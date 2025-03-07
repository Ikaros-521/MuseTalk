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
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
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
import multiprocessing
import signal
import psutil
import json

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

# 在类外部定义进程函数
def process_task_fn(task_id: str, audio_path: str, video_path: str, bbox_shift: float):
    """进程任务处理函数"""
    try:
        if os.name != 'nt':  # 在非Windows系统上设置进程组
            os.setpgrp()
            
        # 确保在新进程中重新初始化 CUDA
        # torch.cuda.empty_cache()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 重新加载模型到当前进程
        # audio_processor, vae, unet, pe = load_all_model()
        # timesteps = torch.tensor([0], device=device)
        
        output_path, bbox_shift_text = inference(
            audio_path,
            video_path,
            bbox_shift
        )
        
        # 将结果写入文件
        result = {
            "status": "completed",
            "output_path": output_path,
            "bbox_shift_text": bbox_shift_text
        }
        with open(os.path.join("temp", task_id, "result.json"), "w") as f:
            json.dump(result, f)
            
        # 清理 GPU 内存
        # torch.cuda.empty_cache()
            
    except Exception as e:
        # 记录错误信息
        result = {
            "status": "failed",
            "error": str(e)
        }
        with open(os.path.join("temp", task_id, "result.json"), "w") as f:
            json.dump(result, f)
        
        # 确保清理 GPU 内存
        torch.cuda.empty_cache()

# 任务管理器
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: List[str] = []
        self.max_concurrent_tasks = 1
        self.current_processing = 0
        self.lock = asyncio.Lock()
        self.processes: Dict[str, multiprocessing.Process] = {}
        self.queue_check_event = asyncio.Event()
        self.processing_tasks = set()  # 新增：跟踪正在处理的任务
    
    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            create_time=datetime.now().isoformat()
        )
        self.tasks[task_id] = task_info
        self.task_queue.append(task_id)
        return task_id
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        return self.tasks.get(task_id)
    
    async def process_queue(self):
        """持续处理队列的后台任务"""
        logger.info("队列处理器开始运行")
        while True:
            try:
                logger.info(f"等待队列事件... (当前队列长度: {len(self.task_queue)}, 正在处理: {self.current_processing})")
                # 重置事件状态
                self.queue_check_event.clear()
                # 等待事件
                await self.queue_check_event.wait()
                
                logger.info("收到队列检查事件，开始处理...")
                logger.info(f"当前队列状态 - 队列长度: {len(self.task_queue)}, 正在处理数: {self.current_processing}")
                
                # 如果当前没有正在处理的任务，并且队列中有任务，则处理下一个任务
                while self.task_queue and self.current_processing < self.max_concurrent_tasks:
                    logger.info("条件满足，准备处理队列中的任务")
                    async with self.lock:
                        try:
                            # 获取并检查下一个任务
                            while self.task_queue:
                                next_task_id = self.task_queue[0]
                                task = self.tasks.get(next_task_id)
                                
                                if not task:
                                    logger.warning(f"任务 {next_task_id} 不存在，从队列中移除")
                                    self.task_queue.pop(0)
                                    continue
                                    
                                if task.status != TaskStatus.PENDING:
                                    logger.warning(f"任务 {next_task_id} 状态不是pending ({task.status})，从队列中移除")
                                    self.task_queue.pop(0)
                                    continue
                                    
                                # 找到有效的pending任务，开始处理
                                break
                            else:
                                # 没有找到有效的任务
                                logger.info("队列中没有有效的pending任务")
                                break
                            
                            logger.info(f"准备处理任务 {next_task_id}")
                            
                            # 获取任务参数
                            temp_dir = os.path.join("temp", next_task_id)
                            params_file = os.path.join(temp_dir, "params.json")
                            
                            if not os.path.exists(temp_dir) or not os.path.exists(params_file):
                                logger.error(f"任务 {next_task_id} 的文件不存在")
                                self.task_queue.pop(0)
                                task.status = TaskStatus.FAILED
                                task.error_message = "任务文件不存在"
                                continue
                                
                            with open(params_file, 'r') as f:
                                params = json.load(f)
                                
                            bbox_shift = params.get("bbox_shift")
                            if bbox_shift is None:
                                logger.error(f"任务 {next_task_id} 的bbox_shift参数未指定")
                                self.task_queue.pop(0)
                                task.status = TaskStatus.FAILED
                                task.error_message = "边界框偏移值未指定"
                                continue
                                
                            # 查找音频和视频文件
                            files = os.listdir(temp_dir)
                            audio_file = next((f for f in files if f.endswith(('.mp3', '.wav'))), None)
                            video_file = next((f for f in files if f.endswith(('.mp4', '.avi', '.mov'))), None)
                            
                            if not audio_file or not video_file:
                                logger.error(f"任务 {next_task_id} 的音频或视频文件不存在")
                                self.task_queue.pop(0)
                                task.status = TaskStatus.FAILED
                                task.error_message = "音频或视频文件不存在"
                                continue
                                
                            audio_path = os.path.join(temp_dir, audio_file)
                            video_path = os.path.join(temp_dir, video_file)
                            
                            # 移除任务从队列并增加处理计数
                            self.task_queue.pop(0)
                            self.current_processing += 1
                            
                            logger.info(f"开始处理任务 {next_task_id}")
                            # 创建新的处理任务
                            asyncio.create_task(self._process_task(
                                next_task_id,
                                audio_path,
                                video_path,
                                bbox_shift
                            ))
                            
                        except Exception as e:
                            logger.error(f"处理队列时出错: {str(e)}")
                            logger.error(traceback.format_exc())
                            if 'next_task_id' in locals() and next_task_id in self.task_queue:
                                self.task_queue.pop(0)
                            if 'task' in locals() and task:
                                task.status = TaskStatus.FAILED
                                task.error_message = str(e)
                else:
                    logger.info(f"无需处理队列 (队列长度: {len(self.task_queue)}, 当前处理数: {self.current_processing})")
                
            except Exception as e:
                logger.error(f"队列处理器出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 短暂等待以避免过于频繁的检查
            await asyncio.sleep(0.1)
    
    async def _process_task(self, task_id: str, audio_path: str, video_path: str, bbox_shift: float):
        """处理单个任务"""
        task = self.tasks[task_id]
        self.processing_tasks.add(task_id)  # 添加到正在处理集合
        try:
            logger.info(f"开始处理任务 {task_id}")
            # 创建新进程运行任务
            process = multiprocessing.Process(
                target=process_task_fn,
                args=(task_id, audio_path, video_path, bbox_shift)
            )
            process.daemon = True
            self.processes[task_id] = process
            process.start()
            
            task.status = TaskStatus.PROCESSING
            task.start_time = datetime.now().isoformat()
            
            # 等待进程完成，每秒检查一次状态
            while process.is_alive():
                # logger.warning(f"任务 {task_id} 仍在运行")
                await asyncio.sleep(1)
            
            logger.info(f"任务 {task_id} 已结束")
            # 进程结束后等待一小段时间确保文件写入完成
            await asyncio.sleep(0.5)
                
            # 检查结果
            result_file = os.path.join("temp", task_id, "result.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    
                if result["status"] == "completed":
                    task.status = TaskStatus.COMPLETED
                    task.result = {
                        "video_name": os.path.basename(result["output_path"]),
                        "video_path": result["output_path"],
                        "bbox_shift_text": result["bbox_shift_text"]
                    }
                    # logger.info(f"任务 {task_id} 完成")
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = result.get("error", "未知错误")
                    logger.error(f"任务 {task_id} 失败: {task.error_message}")
            else:
                task.status = TaskStatus.FAILED
                task.error_message = "任务结果文件不存在"
                logger.error(f"任务 {task_id} 失败: 结果文件不存在")
                
        except Exception as e:
            logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
        finally:
            # 确保进程被正确清理
            if task_id in self.processes:
                process = self.processes[task_id]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                del self.processes[task_id]
            
            task.end_time = datetime.now().isoformat()
            
            async with self.lock:
                if task_id in self.processing_tasks:  # 只有当任务仍在处理集合中时才减少计数
                    self.current_processing -= 1
                    self.processing_tasks.remove(task_id)
                    logger.info(f"任务 {task_id} 处理完成，当前处理数: {self.current_processing}")
                # 触发队列检查
                logger.info("正在触发队列检查事件...")
                self.queue_check_event.set()
                logger.info("队列检查事件已触发")
    
    def _kill_process_tree(self, pid: int):
        """跨平台终止进程树"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], check=False)
            else:  # Linux/Unix
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                
                # 先终止子进程
                for child in children:
                    try:
                        os.kill(child.pid, signal.SIGKILL)
                    except (ProcessLookupError, psutil.NoSuchProcess):
                        pass
                
                # 终止主进程
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
    
    async def stop_task(self, task_id: str) -> bool:
        """跨平台强制停止任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False
            
        if task.status == TaskStatus.PROCESSING:
            # 获取进程对象
            process = self.processes.get(task_id)
            if process and process.is_alive():
                # 终止进程树
                self._kill_process_tree(process.pid)
                
                # 等待进程结束（设置较短的超时时间）
                try:
                    process.join(timeout=2)
                except:
                    pass
                    
            # 从进程字典中移除
            if task_id in self.processes:
                del self.processes[task_id]
                
            task.status = TaskStatus.STOPPED
            task.end_time = datetime.now().isoformat()
            
            async with self.lock:
                self.current_processing -= 1
            
            return True
            
        elif task.status == TaskStatus.PENDING:
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            task.status = TaskStatus.STOPPED
            task.end_time = datetime.now().isoformat()
            return True
            
        return False
    
    async def delete_task(self, task_id: str) -> bool:
        """跨平台删除任务及其所有资源"""
        try:
            # 先停止任务
            await self.stop_task(task_id)
            
            # 删除所有相关文件和记录
            for dir_path in ["temp", "results/output", "results/input"]:
                task_dir = os.path.join(dir_path, task_id)
                if os.path.exists(task_dir):
                    try:
                        shutil.rmtree(task_dir, ignore_errors=True)
                    except:
                        pass
                    
            async with self.lock:
                # 清理任务记录
                if task_id in self.tasks:
                    del self.tasks[task_id]
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
                if task_id in self.processes:
                    del self.processes[task_id]
                if task_id in self.processing_tasks:  # 如果任务正在处理中，更新计数
                    self.processing_tasks.remove(task_id)
                    self.current_processing = max(0, self.current_processing - 1)  # 确保不会小于0
                    
            return True
            
        except Exception as e:
            logger.error(f"清理任务{task_id}资源时出错: {str(e)}")
            return False

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
    try:
        logger.info(f"开始处理任务 - 音频: {audio_path}, 视频: {video_path}, bbox偏移: {bbox_shift}")
        
        args_dict = {"result_dir": './results/output', "fps": 25, "batch_size": 8, "output_vid_name": '', "use_saved_coord": False}
        args = Namespace(**args_dict)

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename = os.path.basename(audio_path).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        output_basename = f"{input_basename}_{audio_basename}_{timestamp}"
        result_img_save_path = os.path.join(args.result_dir, output_basename)
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl")
        os.makedirs(result_img_save_path, exist_ok=True)
        logger.info(f"创建输出目录: {result_img_save_path}")

        if args.output_vid_name == "":
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        logger.info(f"输出视频路径: {output_vid_name}")

        ############################################## extract frames from source video ##############################################
        logger.info("开始提取视频帧...")
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            reader = imageio.get_reader(video_path)
            total_frames = reader.count_frames()
            logger.info(f"视频总帧数: {total_frames}")

            for i, im in enumerate(reader):
                imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
                if i % 100 == 0:  # 每100帧记录一次进度
                    logger.info(f"已处理 {i}/{total_frames} 帧")
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
            logger.info(f"视频帧率: {fps}")
        else:
            logger.info("输入为图片文件夹")
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        logger.info(f"总共处理 {len(input_img_list)} 张图片")

        ############################################## extract audio feature ##############################################
        logger.info("开始提取音频特征...")
        whisper_feature = audio_processor.audio2feat(audio_path)
        logger.info(f"音频特征shape: {whisper_feature.shape}")
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        logger.info(f"音频分块数量: {len(whisper_chunks)}")

        ############################################## preprocess input image  ##############################################
        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
            logger.info("使用已保存的坐标信息")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            logger.info("开始提取人脸特征点...")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
            logger.info("人脸特征点提取完成")
        
        bbox_shift_text=get_bbox_range(input_img_list, bbox_shift)
        logger.info(f"边界框范围: {bbox_shift_text}")

        logger.info("开始处理图像特征...")
        i = 0
        input_latent_list = []
        valid_frames = 0
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
            valid_frames += 1
            if valid_frames % 100 == 0:
                logger.info(f"已处理 {valid_frames} 个有效帧")

        logger.info(f"总共处理 {valid_frames} 个有效帧")

        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        logger.info("完成图像特征处理")

        ############################################## inference batch by batch ##############################################
        logger.info("开始批量推理...")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        total_batches = int(np.ceil(float(video_num)/batch_size))
        logger.info(f"总批次数: {total_batches}, 批次大小: {batch_size}")
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        
        try:
            for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=total_batches)):
                if should_stop and should_stop():
                    logger.info("收到停止信号，中断处理")
                    raise InterruptedError("Task was stopped")
                
                if i % 50 == 0:  # 降低日志记录频率
                    logger.info(f"推理进度: {i}/{total_batches} 批次")
                    # 定期清理 GPU 内存
                    torch.cuda.empty_cache()
                    
                tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
                audio_feature_batch = torch.stack(tensor_list).to(device)
                audio_feature_batch = pe(audio_feature_batch)
                
                with torch.cuda.amp.autocast():  # 使用自动混合精度
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents)
                
                # 添加调试日志
                logger.info(f"recon type: {type(recon)}, first element type: {type(recon[0])}")
                
                # 如果recon是torch.Tensor，转换为numpy数组
                if isinstance(recon[0], torch.Tensor):
                    logger.info("Converting tensors to numpy arrays...")
                    recon = [frame.cpu().numpy() for frame in recon]
                
                res_frame_list.extend(recon)
                
                if i == 0:  # 只在第一批次记录
                    logger.info(f"First batch result shape: {recon[0].shape}")
                
                # 清理临时变量
                del tensor_list, audio_feature_batch, pred_latents
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"批量推理过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 确保清理 GPU 内存
            torch.cuda.empty_cache()
            raise e
        
        logger.info("批量推理完成")
            
        ############################################## pad to full image ##############################################
        logger.info("开始合成最终视频帧...")
        total_frames = len(res_frame_list)
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            if i % 100 == 0:  # 每100帧记录一次进度
                logger.info(f"正在处理第 {i}/{total_frames} 帧")
                
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except Exception as e:
                logger.error(f"处理第 {i} 帧时出错: {str(e)}")
                logger.error(f"bbox: {bbox}")
                logger.error(traceback.format_exc())
                continue
            
            combine_frame = get_image(ori_frame,res_frame,bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)

        logger.info("开始生成视频文件...")
        output_video = 'temp.mp4'

        # 读取图片
        def is_valid_image(file):
            pattern = re.compile(r'\d{8}\.png')
            return pattern.match(file)

        images = []
        files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
        files.sort(key=lambda x: int(x.split('.')[0]))
        logger.info(f"找到 {len(files)} 个有效图片文件")

        for file in files:
            filename = os.path.join(result_img_save_path, file)
            images.append(imageio.imread(filename))

        logger.info("开始编码视频...")
        imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
        logger.info("视频编码完成")

        input_video = './temp.mp4'
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"临时视频文件未找到: {input_video}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件未找到: {audio_path}")
        
        logger.info("开始合成音视频...")
        reader = imageio.get_reader(input_video)
        fps = reader.get_meta_data()['fps']
        reader.close()
        frames = images

        logger.info(f"总帧数: {len(frames)}")

        video_clip = VideoFileClip(input_video)
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)

        logger.info("开始写入最终视频文件...")
        video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)
        logger.info("视频文件写入完成")

        os.remove("temp.mp4")
        logger.info(f"处理完成，结果保存至: {output_vid_name}")
        return output_vid_name,bbox_shift_text
        
    except Exception as e:
        logger.error("处理过程中发生错误:")
        logger.error(traceback.format_exc())
        raise e


app = FastAPI(
    title="MuseTalk API",
    description="视频口型同步API接口",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化函数"""
    logger.info("正在初始化应用...")
    # 确保队列处理器启动
    logger.info("启动队列处理器...")
    # 立即触发一次队列检查
    task_manager.queue_check_event.set()
    # 创建后台任务
    asyncio.create_task(task_manager.process_queue())
    logger.info("队列处理器启动完成")

class VideoResponse(BaseModel):
    video_name: str
    video_path: str
    bbox_shift_text: str

# 在TaskInfo和TaskResponse类定义后添加新的响应模型
class ErrorResponse(BaseModel):
    error: str

@app.post("/tasks/create", response_model=TaskResponse)
async def create_synthesis_task(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="驱动音频文件"),
    video: UploadFile = File(..., description="参考视频文件"),
    bbox_shift: float = Form(0.0, description="边界框偏移值(像素)")
):
    try:
        # 创建任务
        task_id = task_manager.create_task()
        logger.info(f"创建新任务: {task_id}")
        
        # 创建临时目录存储上传的文件
        temp_dir = os.path.join("temp", task_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存上传的文件
        audio_path = os.path.join(temp_dir, audio.filename)
        video_path = os.path.join(temp_dir, video.filename)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        # 保存任务参数
        params = {
            "bbox_shift": bbox_shift
        }
        with open(os.path.join(temp_dir, "params.json"), "w") as f:
            json.dump(params, f)
            
        processed_video_path = check_video(video_path)
        
        # 使用锁检查当前处理任务数量
        async with task_manager.lock:
            if task_manager.current_processing < task_manager.max_concurrent_tasks:
                # 如果未达到最大并发数，直接处理
                logger.info(f"直接处理任务 {task_id}")
                task_manager.current_processing += 1
                # 创建异步任务
                asyncio.create_task(
                    task_manager._process_task(
                        task_id,
                        audio_path,
                        processed_video_path,
                        bbox_shift
                    )
                )
                message = "任务创建成功，正在处理"
            else:
                # 否则加入队列
                logger.info(f"任务 {task_id} 加入队列")
                task_manager.task_queue.append(task_id)
                # 触发队列检查
                task_manager.queue_check_event.set()
                message = "任务创建成功，已加入队列"
        
        return TaskResponse(
            task_id=task_id,
            message=message
        )
        
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        logger.error(traceback.format_exc())
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return TaskResponse(
            task_id=str(uuid.uuid4()),
            message=f"任务创建失败：{str(e)}"
        )

@app.get("/tasks/{task_id}", response_model=TaskInfo, responses={404: {"model": TaskInfo}})
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
        # 返回一个空的TaskInfo对象，但包含错误信息
        return TaskInfo(
            task_id=task_id,
            status=TaskStatus.FAILED,
            create_time=datetime.now().isoformat(),
            error_message="任务不存在"
        )
    return task_info

@app.get("/tasks/{task_id}/download", responses={404: {"model": ErrorResponse}})
async def download_task_result(task_id: str):
    """
    下载任务结果视频
    
    参数:
    - task_id: 任务ID
    """
    task_info = task_manager.get_task_info(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="任务不存在")
        
    if task_info.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")
        
    video_path = task_info.result.get("video_path")  # 使用 get() 方法更安全
    if not video_path:
        raise HTTPException(status_code=404, detail="视频路径未找到")
        
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path)
        )
    raise HTTPException(status_code=404, detail="视频文件不存在")

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
        return TaskResponse(
            task_id=task_id,
            message="任务停止指令已发送"
        )
    return TaskResponse(
        task_id=task_id,
        message="任务不存在或无法停止"
    )

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    删除指定的任务及其相关数据
    
    参数:
    - task_id: 任务ID
    """
    success = await task_manager.delete_task(task_id)
    if success:
        return TaskResponse(
            task_id=task_id,
            message="任务及相关数据已删除"
        )
    return TaskResponse(
        task_id=task_id,
        message="任务不存在或删除失败"
    )

if __name__ == "__main__":
    # 下载模型
    download_model()
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        loop="asyncio"
    ) 