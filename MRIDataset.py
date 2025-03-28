import os
import glob
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pandas as pd

def find_mri_file(patient_folder, mri_pattern="*T1*.hdr"):
    """
    在患者文件夹中查找符合模式的 MRI 文件（hdr文件），
    如果找到多个，则返回第一个匹配的文件。
    """
    files = glob.glob(os.path.join(patient_folder, mri_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No MRI file matching pattern '{mri_pattern}' found in {patient_folder}")
    # 返回第一个匹配的文件
    return files[0]

class MRIDataset(Dataset):
    def __init__(self, root_dir, label_csv, transform=None, mri_pattern="*T1*.hdr"):
        """
        Args:
            root_dir (str): 存放各位患者数据的根目录，每个子文件夹对应一位患者
            label_csv (str): 标签表格文件路径，表格中必须包含“姓名”和“手术名称”两列
            transform (callable, optional): 对数据进行转换的函数
            mri_pattern (str): 用于查找 MRI 文件的模式，默认匹配含 "T1" 的 hdr 文件
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mri_pattern = mri_pattern

        # 遍历所有患者文件夹，假设文件夹名为患者姓名
        self.patient_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                                if os.path.isdir(os.path.join(root_dir, d))]
        # 读取 CSV 标签文件，注意编码可根据实际情况调整，如 utf-8 或 gbk
        self.labels_df = pd.read_csv(label_csv, encoding="utf-8")
        # 构建姓名 -> 标签 的字典
        # 规定：手术名称为"难治性外科手术"或"难治性内科治疗"的为 0 (药物难治型)，
        #       手术名称为"药物治疗"的为 1 (药物敏感型)
        self.label_dict = {}
        for idx, row in self.labels_df.iterrows():
            name = str(row["姓名"]).strip()
            surgery = str(row["手术名称"]).strip()
            if surgery in ["难治性外科手术", "难治性内科治疗"]:
                label = 0
            elif surgery == "药物治疗":
                label = 1
            else:
                label = -1  # 或者根据需要设定默认值
            self.label_dict[name] = label

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]
        patient_name = os.path.basename(patient_folder).strip()

        # 动态查找符合条件的 MRI 文件
        hdr_path = find_mri_file(patient_folder, self.mri_pattern)
        # 加载对应的 MRI 数据
        img_obj = nib.load(hdr_path)
        data = img_obj.get_fdata()

        # 转换为 torch tensor（float类型）
        data = torch.from_numpy(data).float()
        # 如果数据是3D (D,H,W)，增加 channel 维度，变为 (1,D,H,W)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        # 如果数据已经是 4D，但通道维度在最后，则转置为 (1, D, H, W)
        elif len(data.shape) == 4:
            if data.shape[-1] == 1:
                data = data.permute(3, 0, 1, 2)
        if self.transform:
            data = self.transform(data)

        # 根据患者姓名查找标签
        if patient_name in self.label_dict:
            label = self.label_dict[patient_name]
        else:
            raise ValueError(f"Label for patient {patient_name} not found in CSV.")

        label = torch.tensor(label, dtype=torch.long)
        return data, label
