# beam-transfer

### 使用方式

启动类：cli/BeamTransfer.py

```shell
python BeamTransfer.py -a "...\beam-transfer\pic\try\1006.png" -b "...\beam-transfer\pic\try\1008.png" -o "...\beam-transfer\pic\out\1006_1008.png"
```

启动参数已设置默认值，可根据自己的电脑路径进行修改

## labelme

用来给图片打标签，制作自己的数据集

### 使用步骤

1. 在labelme上给每张图片选好目标区域打标签，保存json文件
2. 命令行执行，生成包含标签图的文件夹`xxx_json`，内含`label.png`和`label_viz.png`等
   ```shell
   labelme_json_to_dataset.exe xxx.json
   ```
3. 将`label.png`复制到labels目录中