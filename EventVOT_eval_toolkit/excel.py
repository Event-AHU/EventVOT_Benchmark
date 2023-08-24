import os  
import excel

# 设置工作簿和工作表  
workbook = excel.Workbook()  
worksheet = workbook.active

# 设置要查找的文件路径  
video_folder = r"E:\Event_frame2"

# 遍历文件夹  
for classname in os.listdir(video_folder):  
    class_path = os.path.join(video_folder,classname)
    for filename in class_path : 
        # 构造数据列表  
        data = [filename]  
        worksheet.cell(row=0, column=1, value=data[0])  
        workbook.save(r"E:\video.xlsx") 