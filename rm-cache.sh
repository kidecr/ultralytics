#!/bin/zsh 
  
# 检查参数数量  
if [ "$#" -ne 1 ]; then  
    echo "Usage: $0 <path_to_directory>"  
    exit 1  
fi  
  
# 检查路径是否存在  
if [ ! -d "$1" ]; then  
    echo "Error: The path '$1' does not exist or is not a directory."  
    exit 1  
fi  
  
# 使用find命令查找并删除所有以cache结尾的文件，包括子目录
find "$1" -type f -name "*.cache" -delete -print
