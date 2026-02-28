#!/bin/bash
# Set up Vulkan SDK environment
export VULKAN_SDK=$(pwd)/1.4.321.0/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

# Run the application
# Using the pre-built binary and the flowers_1.ply input
echo "Running 3DGS Renderer..."
python main.py --ply vk_gaussian_splatting/_downloaded_resources/flowers_1/flowers_1.ply --binary vk_gaussian_splatting/_bin/vk_gaussian_splatting
