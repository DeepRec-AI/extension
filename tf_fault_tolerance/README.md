# TF Fault Tolerance
## 简介
**TF-Fault-Tolerance** 通过在PS中缓存checkpoint文件的方式，提升分布式训练任务发生异常时模型恢复效率，快速恢复训练任务。

## 编译本项目
1. 准备一个已经预先安装tensorflow和bazel的镜像。
2. 使用镜像创建并启动一个容器，并进入到项目的根文件夹下。
3. 编译。
```bash
make -j$(nproc) tft
```
4. 生成的whl包位于`tf_fault_tolerance/dist`文件夹内。

## 使用方式
### 前置条件
训练集群需要打开重启失败节点的功能，并且训练镜像中已安装本项目whl包。

### 启用
引用本项目python包，并且将`AsyncCacheCKPTRunnerHook`加入到`chief_only_hooks`即可。

```python
import tensorflow
import tf_fault_tolerance as ft

...

chief_only_hooks = [ft.train.AsyncCacheCKPTRunnerHook()]
with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=is_chief,
        checkpoint_dir="./model_ckpt/",
        save_checkpoint_steps = 100,
        chief_only_hooks=chief_only_hooks,
        hooks=hooks,
        scaffold=scaffold,
        config=sess_config) as sess:
    sess.run(...)
```
