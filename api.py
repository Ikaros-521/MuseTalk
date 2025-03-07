import os
import time
import re
import gradio as gr
import numpy as np
import sys
import subprocess
import requests
import argparse
from omegaconf import OmegaConf
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil
import imageio
import ffmpeg
from moviepy.editor import *
from huggingface_hub import snapshot_download
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
from typing import Optional, Dict, List
from pathlib import Path
import uvicorn
from enum import Enum
import uuid
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import traceback

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder,get_bbox_range
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model

logger.add("api.log", rotation="200MB", retention="10 days", level="INFO")

# 全局变量定义
ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

audio_processor,vae,unet,pe  = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

# 任务状态枚举
class TaskStatus(str, Enum):
    PENDING = "pending"      # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 完成
    FAILED = "failed"         # 失败
    STOPPING = "stopping"     # 正在停止
    STOPPED = "stopped"      # 已停止

# 任务信息模型
class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    create_time: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict] = None

# 任务响应模型
class TaskResponse(BaseModel):
    task_id: str
    message: str

# 任务管理器
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: List[str] = []
        self.max_concurrent_tasks = 2
        self.current_processing = 0
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.stop_flags: Dict[str, bool] = {}
        
    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            create_time=datetime.now().isoformat()
        )
        self.tasks[task_id] = task_info
        self.task_queue.append(task_id)
        self.stop_flags[task_id] = False
        return task_id
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        return self.tasks.get(task_id)
    
    async def stop_task(self, task_id: str) -> bool:
        """停止指定任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False
            
        if task.status == TaskStatus.PROCESSING:
            # 设置停止标志
            self.stop_flags[task_id] = True
            task.status = TaskStatus.STOPPING
            return True
        elif task.status == TaskStatus.PENDING:
            # 如果任务还在队列中，直接移除
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            task.status = TaskStatus.STOPPED
            task.end_time = datetime.now().isoformat()
            return True
        return False
    
    async def delete_task(self, task_id: str) -> bool:
        """删除指定任务及其相关数据"""
        task = self.tasks.get(task_id)
        if not task:
            return False
            
        # 如果任务正在处理，先停止它
        if task.status == TaskStatus.PROCESSING:
            await self.stop_task(task_id)
            # 等待任务真正停止
            for _ in range(10):  # 最多等待10秒
                if task.status not in [TaskStatus.PROCESSING, TaskStatus.STOPPING]:
                    break
                await asyncio.sleep(1)
        
        # 删除相关文件
        try:
            if task.result and "video_path" in task.result:
                video_path = task.result["video_path"]
                if os.path.exists(video_path):
                    os.remove(video_path)
                # 删除相关的临时文件夹
                result_dir = os.path.dirname(video_path)
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
        except Exception as e:
            logger.error(f"删除任务{task_id}的文件时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 从任务队列和任务字典中删除
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        del self.tasks[task_id]
        if task_id in self.stop_flags:
            del self.stop_flags[task_id]
        
        return True
    
    async def process_task(self, task_id: str, audio_path: str, video_path: str, bbox_shift: float):
        async with self.lock:
            if self.current_processing >= self.max_concurrent_tasks:
                return
            self.current_processing += 1
        
        try:
            task = self.tasks[task_id]
            task.status = TaskStatus.PROCESSING
            task.start_time = datetime.now().isoformat()
            
            # 检查是否需要停止
            if self.stop_flags.get(task_id, False):
                task.status = TaskStatus.STOPPED
                task.end_time = datetime.now().isoformat()
                return
            
            # 执行视频合成
            output_path, bbox_shift_text = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                inference,
                audio_path,
                video_path,
                bbox_shift,
                lambda: self.stop_flags.get(task_id, False)  # 传入停止检查函数
            )
            
            # 再次检查是否需要停止
            if self.stop_flags.get(task_id, False):
                task.status = TaskStatus.STOPPED
                task.end_time = datetime.now().isoformat()
                return
                
            task.status = TaskStatus.COMPLETED
            task.result = {
                "video_name": os.path.basename(output_path),
                "video_path": output_path,
                "bbox_shift_text": bbox_shift_text
            }
            
        except Exception as e:
            logger.error(traceback.format_exc())
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
        finally:
            task.end_time = datetime.now().isoformat()
            async with self.lock:
                self.current_processing -= 1
                await self.check_queue()
    
    async def check_queue(self):
        if not self.task_queue:
            return
            
        async with self.lock:
            if self.current_processing >= self.max_concurrent_tasks:
                return
                
            next_task_id = self.task_queue.pop(0)
            task = self.tasks[next_task_id]
            if task.status == TaskStatus.PENDING:
                # 获取任务相关的文件路径和参数
                # 这里需要维护任务的输入参数
                pass

# 创建任务管理器实例
task_manager = TaskManager()

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            logger.info(child_path)

def download_model():
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        logger.info("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # 其他模型下载代码...
        toc = time.time()
        logger.info(f"download cost {toc-tic} seconds")
        print_directory_contents(CheckpointsDir)
    else:
        logger.info("Already download the model.")

def get_file_type(path):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    ext = os.path.splitext(path)[1].lower()
    return "video" if ext in video_formats else "image"

def get_video_fps(video_path):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    reader.close()
    return fps

def check_video(video):
    if not isinstance(video, str):
        return video
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/output', exist_ok=True)
    os.makedirs('./results/input', exist_ok=True)

    output_video = os.path.join('./results/input', output_file_name)

    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']

    frames = [im for im in reader]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, should_stop=None, progress=None):
    args_dict = {"result_dir": './results/output', "fps": 25, "batch_size": 8, "output_vid_name": '', "use_saved_coord": False}
    args = Namespace(**args_dict)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    # 文件名追加时间戳毫秒
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_basename = f"{input_basename}_{audio_basename}_{timestamp}"
    result_img_save_path = os.path.join(args.result_dir, output_basename)
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)

    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path)=="video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full,exist_ok = True)
        # cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        # os.system(cmd)
        # 读取视频
        reader = imageio.get_reader(video_path)

        # 保存图片
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    #print(input_img_list)
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        logger.info("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        logger.info("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text=get_bbox_range(input_img_list, bbox_shift)
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################
    logger.info("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        # 检查是否需要停止
        if should_stop and should_stop():
            raise InterruptedError("Task was stopped")
            
        tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
        audio_feature_batch = torch.stack(tensor_list).to(unet.device)
        audio_feature_batch = pe(audio_feature_batch)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    logger.info("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"bbox: {bbox}, error: {e}")
            continue
        
        combine_frame = get_image(ori_frame,res_frame,bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p temp.mp4"
    # print(cmd_img2video)
    # os.system(cmd_img2video)
    # 帧率
    fps = 25
    # 图片路径
    # 输出视频路径
    output_video = 'temp.mp4'

    # 读取图片
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # 保存视频
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    # cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {output_vid_name}"
    # print(cmd_combine_audio)
    # os.system(cmd_combine_audio)

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 读取视频
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率
    reader.close() # 否则在win11上会报错：PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。: 'temp.mp4'
    # 将帧存储在列表中
    frames = images

    # 保存视频并添加音频
    # imageio.mimwrite(output_vid_name, frames, 'FFMPEG', fps=fps, codec='libx264', audio_codec='aac', input_params=['-i', audio_path])
    
    # input_video = ffmpeg.input(input_video)
    
    # input_audio = ffmpeg.input(audio_path)
    
    logger.info(len(frames))

    # imageio.mimwrite(
    #     output_video,
    #     frames,
    #     'FFMPEG',
    #     fps=25,
    #     codec='libx264',
    #     audio_codec='aac',
    #     input_params=['-i', audio_path],
    #     output_params=['-y'],  # Add the '-y' flag to overwrite the output file if it exists
    # )
    # writer = imageio.get_writer(output_vid_name, fps = 25, codec='libx264', quality=10, pixelformat='yuvj444p')
    # for im in frames:
    #     writer.append_data(im)
    # writer.close()




    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)

    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    logger.info(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text


app = FastAPI(
    title="MuseTalk API",
    description="视频口型同步API接口",
    version="1.0.0"
)

class VideoResponse(BaseModel):
    video_name: str
    video_path: str
    bbox_shift_text: str

@app.post("/tasks/create", response_model=TaskResponse)
async def create_synthesis_task(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="驱动音频文件"),
    video: UploadFile = File(..., description="参考视频文件"),
    bbox_shift: float = Form(0.0, description="边界框偏移值(像素)")
):
    """
    创建视频合成任务
    
    参数:
    - audio: 驱动音频文件
    - video: 参考视频文件
    - bbox_shift: 边界框偏移值，默认为0
    
    返回:
    - task_id: 任务ID
    - message: 任务创建状态信息
    """
    try:
        # 创建临时目录存储上传的文件
        temp_dir = os.path.join("temp", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存上传的文件
        audio_path = os.path.join(temp_dir, audio.filename)
        video_path = os.path.join(temp_dir, video.filename)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        processed_video_path = check_video(video_path)
        
        # 创建任务
        task_id = task_manager.create_task()
        
        # 添加到后台任务
        background_tasks.add_task(
            task_manager.process_task,
            task_id,
            audio_path,
            processed_video_path,
            bbox_shift
        )
        
        return TaskResponse(
            task_id=task_id,
            message="任务创建成功，正在排队处理"
        )
        
    except Exception as e:
        logger.error(traceback.format_exc())
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {"error": str(e)}

@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    参数:
    - task_id: 任务ID
    
    返回:
    - TaskInfo: 任务详细信息
    """
    task_info = task_manager.get_task_info(task_id)
    if task_info is None:
        return {"error": "任务不存在"}
    return task_info

@app.get("/tasks/{task_id}/download")
async def download_task_result(task_id: str):
    """
    下载任务结果视频
    
    参数:
    - task_id: 任务ID
    """
    task_info = task_manager.get_task_info(task_id)
    if task_info is None:
        return {"error": "任务不存在"}
        
    if task_info.status != TaskStatus.COMPLETED:
        return {"error": "任务尚未完成"}
        
    video_path = task_info.result["video_path"]
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path)
        )
    return {"error": "视频文件不存在"}

@app.get("/tasks")
async def list_tasks():
    """
    获取所有任务列表
    
    返回:
    - tasks: 任务列表
    """
    return {"tasks": list(task_manager.tasks.values())}

@app.post("/synthesize/", response_model=VideoResponse)
async def synthesize_video(
    audio: UploadFile = File(..., description="驱动音频文件"),
    video: UploadFile = File(..., description="参考视频文件"),
    bbox_shift: float = Form(0.0, description="边界框偏移值(像素)")
):
    """
    合成口型同步视频
    
    参数:
    - audio: 驱动音频文件
    - video: 参考视频文件
    - bbox_shift: 边界框偏移值，默认为0
    
    返回:
    - video_path: 生成的视频文件路径
    - bbox_shift_text: 边界框范围信息
    """
    try:
        temp_dir = tempfile.mkdtemp()
        
        audio_path = os.path.join(temp_dir, audio.filename)
        video_path = os.path.join(temp_dir, video.filename)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        processed_video_path = check_video(video_path)
        
        output_path, bbox_shift_text = inference(
            audio_path=audio_path,
            video_path=processed_video_path,
            bbox_shift=bbox_shift
        )
        
        return VideoResponse(
            video_name=os.path.basename(output_path),
            video_path=output_path,
            bbox_shift_text=bbox_shift_text
        )
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/download/{video_name}")
async def download_video(video_name: str):
    """
    下载生成的视频文件
    
    参数:
    - video_name: 视频文件名
    """
    video_path = os.path.join("results", "output", video_name)
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_name
        )
    return {"error": "视频文件不存在"}

@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    """
    停止指定的任务
    
    参数:
    - task_id: 任务ID
    """
    success = await task_manager.stop_task(task_id)
    if success:
        return {"message": "任务停止指令已发送"}
    return {"error": "任务不存在或无法停止"}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    删除指定的任务及其相关数据
    
    参数:
    - task_id: 任务ID
    """
    success = await task_manager.delete_task(task_id)
    if success:
        return {"message": "任务及相关数据已删除"}
    return {"error": "任务不存在或删除失败"}

if __name__ == "__main__":
    # 下载模型
    download_model()
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 