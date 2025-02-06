import os
import numpy as np
import nibabel as nib
import natsort
import matplotlib.pyplot as plt


lacune_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort_rlk_raw/with_lacune_prep'

flair_path = os.path.join(lacune_path, "flair")
t1_path = os.path.join(lacune_path, "t1")
t2_path = os.path.join(lacune_path, "t2")
label_path = os.path.join(lacune_path, "Rater")

total_flair_list = natsort.natsorted(os.listdir(flair_path))
total_t1_list = natsort.natsorted(os.listdir(t1_path))
total_t2_list = natsort.natsorted(os.listdir(t2_path))
total_label_list = natsort.natsorted(os.listdir(label_path))

intensity_diffs = {"subject": [], "flair": [], "t1": [], "flair_bg": [], "t1_bg": [], "flair_diff": [], "t1_diff": []}

for i in range(10):
    
    flair = nib.load(flair_path + '/' + total_flair_list[i+6]).get_fdata()
    t1 = nib.load(t1_path + '/' + total_t1_list[i+6]).get_fdata()
    label = nib.load(label_path + '/' + total_label_list[i+6]).get_fdata()

    lacune_mask = label > 0
    non_lacune_mask = label == 0

    flair_intensity = flair[lacune_mask]
    t1_intensity = t1[lacune_mask]

    flair_bg_intensity = flair[non_lacune_mask]
    t1_bg_intensity = t1[non_lacune_mask]

    flair_intensity_mean = np.mean(flair_intensity)
    flair_bg_intensity_mean = np.mean(flair_bg_intensity)
    flair_intensity_mean_diff = flair_intensity_mean - flair_bg_intensity_mean

    t1_intensity_mean = np.mean(t1_intensity)
    t1_bg_intensity_mean = np.mean(t1_bg_intensity)
    t1_intensity_mean_diff = t1_intensity_mean - t1_bg_intensity_mean

    intensity_diffs["subject"].append(f"Subject {i+1}")
    intensity_diffs["flair"].append(flair_intensity_mean)
    intensity_diffs["t1"].append(t1_intensity_mean)
    intensity_diffs["flair_bg"].append(flair_bg_intensity_mean)
    intensity_diffs["t1_bg"].append(t1_bg_intensity_mean)
    intensity_diffs["flair_diff"].append(flair_intensity_mean_diff)
    intensity_diffs["t1_diff"].append(t1_intensity_mean_diff)

# for i in range(len(total_label_list)):
    
#     # flair = nib.load(flair_path + '/' + total_flair_list[i]).get_fdata()
#     # t1 = nib.load(t1_path + '/' + total_t1_list[i]).get_fdata()
#     # t2 = nib.load(t2_path + '/' + total_t2_list[i]).get_fdata()
#     label = nib.load(label_path + '/' + total_label_list[i]).get_fdata()
    
#     for j in range(label.shape[2]):
        
#         label_slice = label[:, :, j]
#         if np.max(label_slice) > 0.0:
#             print("{} : slice = {}".format(total_label_list[i][0:7], str(j)))
    
#     lacune_mask = label > 0
#     non_lacune_mask = label == 0
    
#     flair_intensity = flair[lacune_mask]
#     t1_intensity = t1[lacune_mask]
#     t2_intensity = t2[lacune_mask]
    
#     flair_bg_intensity = flair[non_lacune_mask]
#     t1_bg_intensity = t1[non_lacune_mask]
#     t2_bg_intensity = t2[non_lacune_mask]
    
#     flair_intensity_mean = np.mean(flair_intensity)
#     flair_bg_intensity_mean = np.mean(flair_bg_intensity)
#     flair_intensity_mean_diff = flair_intensity_mean - flair_bg_intensity_mean
    
#     t1_intensity_mean = np.mean(t1_intensity)
#     t1_bg_intensity_mean = np.mean(t1_bg_intensity)
#     t1_intensity_mean_diff = t1_intensity_mean - t1_bg_intensity_mean
    
#     t2_intensity_mean = np.mean(t2_intensity)
#     t2_bg_intensity_mean = np.mean(t2_bg_intensity)
#     t2_intensity_mean_diff = t2_intensity_mean - t2_bg_intensity_mean
    
#     intensity_diffs["subject"].append(f"{i+1}")
#     intensity_diffs["flair"].append(flair_intensity_mean_diff)
#     intensity_diffs["t1"].append(t1_intensity_mean_diff)
#     intensity_diffs["t2"].append(t2_intensity_mean_diff)
    
#     print(f"Subject {i+1} - FLAIR: Lacune area mean = {flair_intensity_mean:.3f} and Background area mean = {flair_bg_intensity_mean:.3f}")
#     print(f"Subject {i+1} - T1: Lacune area mean = {t1_intensity_mean:.3f} and Background area mean = {t1_bg_intensity_mean:.3f}")
#     print(f"Subject {i+1} - T2: Lacune area mean = {t2_intensity_mean:.3f} and Background area mean = {t2_bg_intensity_mean:.3f}")
#     print("")

# # 시각화
fig, ax = plt.subplots(1, 2, figsize=(15, 10))

x = np.arange(len(intensity_diffs["subject"]))
width = 0.3

# # 각 modality 별 평균 intensity 차이 막대 그래프
ax[0].bar(x - width, intensity_diffs["flair"], width=width, color='red', label="Lacune")
ax[0].bar(x, intensity_diffs["flair_bg"], width=width, color='blue', label="Background")
ax[0].bar(x + width, intensity_diffs["flair_diff"], width=width, color='green', label="Difference")
ax[0].set_title("FLAIR Lacune and Background Intensity")
ax[0].set_xlabel("FLAIR")
ax[0].set_ylabel("Intensity")
ax[0].set_xticks(x)
ax[0].set_xticklabels(intensity_diffs["subject"], rotation=45)
ax[0].legend()

ax[1].bar(x - width, intensity_diffs["t1"], width=width, color='red', label="Lacune")
ax[1].bar(x, intensity_diffs["t1_bg"], width=width, color='blue', label="Background")
ax[1].bar(x + width, intensity_diffs["t1_diff"], width=width, color='green', label="Difference")
ax[1].set_title("T1 Lacune and Background Intensity")
ax[1].set_xlabel("T1")
ax[1].set_ylabel("Intensity")
ax[1].set_xticks(x)
ax[1].set_xticklabels(intensity_diffs["subject"], rotation=45)
ax[1].legend()

# ax[2].bar(intensity_diffs["subject"], intensity_diffs["t2"], color='green')
# ax[2].set_title("T2 Lacune - Background Intensity Difference")
# ax[2].set_xlabel("Subject")
# ax[2].set_ylabel("Intensity Difference")

plt.tight_layout()
plt.show()

