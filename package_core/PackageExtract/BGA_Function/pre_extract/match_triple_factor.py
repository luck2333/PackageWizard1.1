import os
import numpy as np
from scipy.spatial.distance import cdist
from package_core.PackageExtract.function_tool import find_list


def match_arrow_pairs_with_yolox(L3, image_root):
    """将尺寸箭头框与带方向的尺寸数字进行最近邻匹配。

    每个视图会尝试把 ``pairs_length`` 中的箭头框与 ``yolox_nums_direction``
    的检测结果关联起来，并在结果中加入视图名称与预留的参数字段。
    """

    all_views_results = []
    print("开始执行循环")
    for view in ("top", "bottom", "side", "detailed"):
        pairs_length = find_list(L3, f"{view}_yolox_pairs_length")
        yolox_nums_direction = find_list(L3, f"{view}_yolox_nums_direction")
        img_path = os.path.join(image_root, f"{view}.jpg")
        print(f"{view}方向准备3合一")
        print(f"箭头数据:{pairs_length}")
        print(f"尺寸数据:{yolox_nums_direction}")

        matched_yolox = [item.copy() for item in yolox_nums_direction]

        for item in matched_yolox:
            item['arrow_pairs'] = None

        if len(pairs_length) == 0:
            all_views_results.append([])
            continue

        arrow_centers = []
        for pair in pairs_length:
            x1, y1, x2, y2 = pair[0:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            arrow_centers.append([center_x, center_y])
        arrow_centers = np.array(arrow_centers)

        yolox_centers = []
        for item in matched_yolox:
            loc = item['location']
            if len(loc) == 4:
                x1, y1, x2, y2 = loc
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
            else:
                center_x, center_y = loc[0], loc[1]
            yolox_centers.append([center_x, center_y])
        yolox_centers = np.array(yolox_centers)

        if yolox_centers.size != 0 and arrow_centers.size != 0:
            distance_matrix = cdist(yolox_centers, arrow_centers)
            yolox_matched = [False] * len(matched_yolox)
            arrow_matched = [False] * len(pairs_length)

            while True:
                min_distance = float('inf')
                min_i, min_j = -1, -1

                for i in range(len(matched_yolox)):
                    if yolox_matched[i]:
                        continue
                    for j in range(len(pairs_length)):
                        if arrow_matched[j]:
                            continue
                        if distance_matrix[i, j] < min_distance:
                            min_distance = distance_matrix[i, j]
                            min_i, min_j = i, j

                if min_i == -1 or min_j == -1:
                    break

                yolox_matched[min_i] = True
                arrow_matched[min_j] = True
                matched_yolox[min_i]['arrow_pairs'] = pairs_length[min_j]

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        processed_list = []
        for item in matched_yolox:
            new_item = {
                'view_name': img_name,
                'parameters': None,
                **item,
            }
            processed_list.append(new_item)

        all_views_results.append(processed_list)

    return all_views_results
