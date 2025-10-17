# DRKD mAP 曲线图使用说明

## 脚本
`plot_drkd_comparison.py` —— 支持参数化切换单/双栏布局、线宽、marker、legend 位置等。

## 快速使用
```bash
# 双栏（3×1 竖排，宽度 3.5 in）
python plot_drkd_comparison.py

# 单栏（1×3 横排，宽度 1.68 in）
# 编辑脚本 DOUBLE_COLUMN = False 后重新运行
```

## 主要参数
| 变量名               | 说明                           | 默认值  |
|--------------------|--------------------------------|--------|
| DOUBLE_COLUMN      | True→双栏，False→单栏          | True   |
| FIG_WIDTH_IN / FIG_HEIGHT_IN | 单栏 3.3×2.2 in（默认），可改双栏 | 3.3 / 2.2 |
| SHOW_TEXT_PLACEHOLDER | 是否放占位文字（用于对齐）   | True   |
| LINE_WIDTH         | 线宽                           | 1.2    |
| MARKER_SIZE        | marker 大小                    | 3      |
| LEGEND_LOC         | legend 位置                    | 'lower right' |

### 合并对比图（新）
运行脚本将生成**单图双曲线**对比：
- 蓝色实线：`atto-from-pico`  
- 红色虚线：`n-from-s`  
文件名仍为 `drkd_map_curve_CVPR_style.{png,pdf,svg,eps}`，可直接插入论文。

## 输出文件
`figures/drkd_map_curve_CVPR_style.{png,pdf,svg,eps}`

## 数据说明
- atto-from-pico：24 个 epoch，best 0.3628
- n-from-s：23 个 epoch，best 0.5937

已适配 IEEE/CVPR 论文排版要求，可直接插入 LaTeX 双栏或单栏。