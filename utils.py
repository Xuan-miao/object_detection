import matplotlib.pyplot as plt
import torch


def box_corner_to_center(boxes: torch.tensor) -> torch.tensor:
    """
    将锚框四角坐标转换为中心坐标
    :param boxes: (x1, y1, x2, y2)
    :return: (x, y, w, h)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def box_center_to_corner(boxes: torch.tensor) -> torch.tensor:
    """
    将锚框中心坐标转换为四角坐标
    :param boxes: (x, y, w, h)
    :return: (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox_to_rect(bbox: torch.tensor, color: str):
    """
    将边界框（左上x，左上y，右下x，右下y）格式转换成matplotlib格式：
    （（左上x，左上y）宽，高）
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]),
                         width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
                         fill=False, edgecolor=color, linewidth=2)


def multi_box_prior(data: torch.tensor, scales: list,
                    ratios: list) -> torch.tensor:
    """
    以每个像素为中心生辰不同形状的锚框
    :param data: 图片 (batch_size, channel, width, height)
    :param scales: 缩放比
    :param ratios: 宽高比
    :return: 锚框
    """
    in_height, in_width = data.shape[-2:]
    device, num_scales, num_ratios = data.device, len(scales), len(ratios)
    boxes_per_pixel = (num_scales + num_ratios - 1)
    scale_tensor = torch.tensor(scales, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚框中心点移动到像素的中心，需要设置偏移量
    # 因为1像素的高宽均为1，我们选择偏移量中心为0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h, steps_w = 1.0 / in_height, 1.0 / in_width  # 在x轴，y轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成"boxes_per_pixel"个高和宽
    # 之后用于创建锚框的四角坐标(x_min,y_min,x_max,y_max)
    w = torch.cat((scale_tensor * torch.sqrt(ratio_tensor[0]),
                   scales[0] * torch.sqrt(ratio_tensor[1:])))
    h = torch.cat((scale_tensor / torch.sqrt(ratio_tensor[0]),
                   scales[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """
    计算交并比
    :param boxes1: (boxes1的数量, 4)
    :param boxes2: (boxes2的数量, 4)
    :return: 交并比矩阵 (boxes1的数量, boxes2的数量)
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1, boxes2, area1, area2的形状分别为
    # (boxes1的数量, 4)
    # (boxes2的数量, 4)
    # (boxes1的数量,)
    # (boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upper_lefts, inter_lower_rights, inters的形状为
    # (boxes1的数量, boxes2的数量, 2)
    inter_upper_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lower_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lower_rights - inter_upper_lefts).clamp(min=0)
    # inter_areas, union_areas的形状为(boxes1的数量, boxes2的数量）
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_box(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框,
    :param ground_truth: 真实边界框
    :param anchors: 锚框
    :param device:
    :param iou_threshold:
    :return: 锚框对应的真实边界框的索引 (type: torch.tensor, shape: (锚框数量,))
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配真实的边界框张量（分配真实边界框的索引）
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实的边界框
    max_iou, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_iou >= iou_threshold).reshape(-1)  #
    box_j = indices[max_iou >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    colum_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = colum_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    对锚框偏移量的转换
    :param anchors: 锚框
    :param assigned_bb: 分配边界框
    :param eps:
    :return: 偏移量
    """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset


def multi_box_target(anchors, labels):
    """
    使用真实边界框标注锚框
    :param anchors: 锚框
    :param labels: 真实边界框
    :return:
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_box(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类别标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标注锚框的类别
        # 如果一个锚框没有被分配， 标注其为背景类别（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """
    根据带有预测偏移量的锚框来预测边界框
    :param anchors:
    :param offset_preds:
    :return:
    """
    anc = box_corner_to_center(anchors)
    pred_box_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_box_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_box_xy, pred_box_wh), dim=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """
    对预测边界框的置信度排序
    :param boxes:
    :param scores:
    :param iou_threshold:
    :return:
    """
    b = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while b.numel() > 0:
        i = b[0]
        keep.append(i)
        if b.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[b[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        b = b[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multi_box_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                        pos_threshold=0.009999999):
    """
    使用非极大值抑制来与预测边界框
    :param cls_probs:
    :param offset_preds:
    :param anchors:
    :param nms_threshold:
    :param pos_threshold:
    :return:
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类别设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
