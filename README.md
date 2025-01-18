# MAC_V1: a classifier of close genes between MCR and ACR

## 项目描述
在对氨基酸序列进行基因类型注释时，存在一些无法通过 blastp 区分开的相似基因，这些基因功能相近但不同，我们有必要将其区分。MCR 和 ACR 就是一对儿功能相近的不同基因，前者催化甲烷生成过程，后者催化中长链烷烃降解过程，本项目可以很好地区分这两种基因。本项目使用了 Lenet 作为神经网络框架，使用 blosum62 矩阵对氨基酸序列进行 embedding。模型在训练集和验证集上都表现出了极其优异的性能，预测准确率都接近于 1。这给传统的依靠 blast 计算序列相似性的基因注释软件提供了启示，或许 ”blast + 神经网络“ 模式能大幅度提高注释软件的性能。


## 使用
**1. 下载项目到本地**

	git clone https://github.com/LiuWeiAIinBio/MAC_V1.git

打开上面下载的文件夹，通过终端命令行进入该文件夹：`cd ./MAC_V1`

<br/>

**2. 预测样本**

注意：电脑需要具备相应的 python 环境

	python predict.py --path_AAfasta_seq 氨基酸序列文件路径 --path_label_file 氨基酸序列对应的标签文件的地址

示例：

	python predict.py --path_AAfasta_seq D:\Desktop\MAC_V1\data\acrA_mcrA_AA_sample.txt --path_label_file D:\Desktop\MAC_V1\data\acrA_mcrA_label_sample.txt

<br/>

**3. 数据处理**

氨基酸序列需要处理为 fasta 格式，例如：

	>GCA_002498745.1_0258_1 coenzyme-B sulfoethylthiotransferase subunit alpha
	MAVKHTKKLFVKALNKKFGKDFDLASQKVEYKRLGPEQNARKREFMEYAKKLEGKRGMTG
	YNPYVHAGGIPLGQRQLVPYKLSSTEYVVEGDDLHFVNNPAMQQFWDDIRRTIVVGLDMA
	HEVLQKRLGKEVTPETINNYLEILNHAMPGAAVVQEHMVETHPGLVDDCNVRVFSGDDAL
	ADEIDDQYKIDINKMFPEDQAETLKAAIGKTTWQAIHVPTIVVRTCDGGTTSRWSAMQLC
	MTFIDAYNMCAGEAAVADLAYAAKHAAVLQMSDMLPARRARGPNNPGGISFGFMSDMVQT
	SRVKPQDPVYVSLNVVAAGSMFYDQIWLGSYMSGGVGFTQYATAAYTNDVLDDFSYYGVD
	YANDKFGGFAKAPATIDVAKELATEVTLYGIEQYEAFPTLLEDHFGGSQRAAVLAAASGI
	TSAIATGHSQIGLAGWYLSMLLHKEAWGRLGFFGYDLQDQCGPTNVFSYQSDEGNPLELR
	GANYPNYAMNVGHQGEYAGITSAAHAGRMDAFAVNPLIKVTFANPGLVFDFTNVRDCFGK
	GGAREFRAAGERSLVMPAV*
	>GCA_002499445.1_0037_1 coenzyme-B sulfoethylthiotransferase subunit alpha
	MAADIFSKFKKSMEVKFTQEFGSNQQSGGDISGKTAKFLRLGPEQDPRKVEMIKAGKELA
	EKRGLAFYNPMMHSGAPLGQRAITPYTISGTDIIAEPDDLHYVNNAAMQQMWDDIRRTCI
	VGLDMAHETLEKRLGKEVTPETINHYLEVLNHAMPGAAVVQEMMVETHPALVDDCFVKVF
	TGDDTLADEIDKQFLIDINKEFPEEQAAQIKASIGKTSWQAIHIPTIVSRTTDGAQTSRW
	AAMQIGMSFISAYSMCAGEAAVADLSFAAKHAALVSMGEMLPARRARGPNEPGGLSFGHL
	SDIIQTSRTVLDDPAKVALEVVGAGCMLYDQIWLGSYMSGGVGFTQYATAAYTDDILDNN
	VYYNVDYINDKYNGAANIGTDNKVKATLDVVKDIATESTLYGIETYEKFPTALEDHFGGS
	QRATVLAAAAGVATALATANANAGLSGWYLSMYLHKEAWGRLGFFGYDLQDQCGATNVLS
	YQGDEGLPDELRGPNYPNYAMNVGHQGGYAGIAQAAHSGRGDAFTVNPLIKVCFADDLLP
	FNFAEPRREFGRGAMREFMPAGERSLVIPAK*
	
氨基酸序列对应的标签需要处理为 0-1 格式，MCR 为 0，ACR 为 1，例如：

	0
	1
	0
	0
	0
	1
	0
	1
	1
	0
