# 第一次上传 GitHub：按这份说明做

这个包尚未上传到 GitHub，不需要安装 Git，也不需要输入命令。

## 1. 先找到要上传的内容

解压压缩包，进入 `dance-motion-similarity` 文件夹。

你应该看到 `README.md`、`requirements.txt`、17 个 Python 文件，以及 `docs`、`materials` 文件夹和 `.gitignore`。

**上传这个文件夹里面的内容，不是上传 ZIP 文件，也不是把外层整个文件夹套进仓库。** GitHub 不会自动把 ZIP 展开成项目主页。这份中文说明在项目文件夹外，不需要上传。

## 2. 建立仓库

1. 打开 https://github.com ，注册或登录。
2. 打开 https://github.com/new 。
3. Repository name 填 `dance-motion-similarity`。
4. Description 可以填：`Video-based dance motion comparison using OpenPose, DTW, and joint-angle analysis.`
5. 如果组员还没有确认公开代码，先选 **Private**。获得团队同意后，可以改为 **Public**；只有公开仓库才能让未授权访客直接查看。
6. 不要额外创建 README 或 .gitignore，这个包已经提供。许可证暂不选择，先与组员确定。
7. 点击 **Create repository**。

## 3. 上传文件

1. 新建空仓库的页面，点击 **uploading an existing file**。如果仓库已经有文件，点 **Add file → Upload files**。
2. 在电脑上打开 `dance-motion-similarity` 文件夹，选中其中的文件和 `docs`、`materials` 文件夹，拖到网页上传区。
3. 如果看不到 `.gitignore`，在 Windows 文件资源管理器打开隐藏项目显示，或手动确认它已经出现在上传列表中。
4. 确认网页列表的最外层有 `README.md`，而不是 `dance-motion-similarity/README.md`。
5. 提交说明填 `Add initial project overview and V1.2 research code`。
6. 对自己刚建的空仓库，提交到默认分支即可，点击 **Commit changes**。如果界面显示 **Propose changes**，按提示创建并合并 Pull request；不要只提交到新分支后忘记合并。

不要额外拖入数据集、OpenPose 安装目录、视频、缓存或个人文件。网页手动上传时不要依赖 .gitignore 替你检查文件。

## 4. 检查和获得链接

上传完成后，仓库首页应自动显示英文项目介绍，下方可以打开代码、文档。

仓库链接形式为：`https://github.com/你的用户名/dance-motion-similarity`。

确认仓库是 Public 后，用未登录的浏览器窗口打开链接，检查别人是否能看到。若仓库为 Private，只有你授权的人能访问。

现在还没有实际仓库链接；完成建仓库和上传后才会产生你的项目页面。

## 5. 以后怎么补 Poster、PPT 和视频

- Poster 或导出的 PPT PDF：上传到 `materials`，再编辑 `materials/README.md` 添加链接。
- 示例 Markdown：`[Project poster](poster.pdf)`。只有文件上传后再添加这条链接。
- Demo 视频：先确认素材公开许可，较大的文件使用审核过的外部视频链接。
- 更新项目介绍：打开 `README.md`，点击编辑按钮，修改后提交。
- 仓库不改名、不转移时，补充材料不会改变仓库链接。

## 当前包说明

采用桌面 `v1.0-v1.2` 文件夹中的最新 V1.2（`-1x368`），不是 Downloads 中较早的 `432x240` 版本。

原始 17 个 Python 文件完整保留，没有更改组员算法。新增英文介绍、运行说明、依赖清单、实验索引、数据说明和忽略规则。没有默认授予软件许可证，也没有填入未经确认的作者名单。

代码文件里仍保留 `D:\OpenPose\openpose` 等原有配置示例。真正运行时需修改成自己的路径、另外安装 OpenPose 并提供视频；仅上传 GitHub 不需要完成这些安装。

这次没有运行完整的 OpenPose 提取或大规模验证，也没有把回归脚本中的期望值说成新跑出的结果。

## 官方帮助

- 建仓库：https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- 上传文件：https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

GitHub 官方文档当前说明：网页单个文件上限为 25 MiB，每次最多上传 100 个文件。这个包低于这些限制。
