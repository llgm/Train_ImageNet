## 在自己的模型上训练ImageNet数据集
#### 主要代码来自顶会开源代码，个人做了部分修改方便自己这种菜鸟使用，如有侵权联系本人删除

**cfgs**文件夹是用来放部分模型的配置文件信息的，my_model.yaml是一个示例模板

**data文件夹**是用来加载数据集与分配数据到GPU上的，不需要改直接使用就行

**models文件夹**用来存放自己model文件，并在build.py文件中修改相应的参数

**config.py**是模型参数的全部信息，在训练之前修改相应的参数，比如根据自己的GPU显存大小设置`batch_size`，以及数据集的路径，默认数据集下分好了**train**和**val**两个文件夹，如不同需要自行修改

**main.py**是代码运行的主要函数，其中需要根据你的GPU编号修改部分信息，不太清楚的可以查看我的csdn博客找到相关设置代码（https://blog.csdn.net/lgm2667419972/article/details/132901028?spm=1001.2014.3001.5502）
相比其他大佬的使用代码我的更加简单，相当于写好模板，照搬写编号数字就行

其他的优化器加载数据集代码都不需要改，直接用就行

代码运行命令示例

`python -m torch.distributed.launch --nproc_per_node=4 main.py --cfg cfgs/my_model.yaml`

`--nproc_per_node=4`代表使用4张GPU，GPU使用的编号在main.py文件中设置4个编号，根据自己情况做修改，--cfg参数是使用自己模型参数配置文件的路径
