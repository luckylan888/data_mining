"""按ACL双栏版式生成实验报告（Word + PDF），内容取自实际运行结果。"""
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figs"
OUT_DOCX = ROOT / "兰丽华_数据挖掘实验报告.docx"

EXTRA = json.loads((ROOT / "extra_numbers.json").read_text(encoding="utf-8"))

CN_FONT = "宋体"
CN_HEAD = "黑体"
EN_FONT = "Times New Roman"


def set_run(run, size=10, bold=False, italic=False, latin=EN_FONT, east=CN_FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    run.font.color.rgb = RGBColor(0, 0, 0)


def body(text, size=10, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         before=0, after=0, line=13.5):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(line)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    set_run(r, size)
    return p


def heading(text, size=11, before=8, after=3):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(size + 4)
    r = p.add_run(text)
    set_run(r, size, bold=True, east=CN_HEAD)
    return p


def set_three_line(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for tag, sz in (("top", 8), ("bottom", 8)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for tag in ("left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tblPr.append(borders)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcB = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:color"), "000000")
        tcB.append(bottom)
        tcPr.append(tcB)


def add_table(caption, headers, rows, widths, font=7.8, first_left=True):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(2)
    cap.paragraph_format.keep_with_next = True
    r = cap.add_run(caption)
    set_run(r, font, bold=False)

    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 固定版式：Word在fixed布局下按tblGrid取列宽，必须同时写网格列宽与单元格宽度，
    # 否则表格会按整页宽度均分并溢出栏外。
    tblPr = table._tbl.tblPr
    layout = tblPr.find(qn("w:tblLayout"))
    total_twips = int(round(sum(widths) * 567))
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        layout.addprevious(tblW)
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(total_twips))
    for i, w in enumerate(widths):
        table.columns[i].width = Cm(w)
    # 收窄单元格左右内边距，给数字列留出宽度，避免0.8415被拆成两行
    cell_mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "45")
        el.set(qn("w:type"), "dxa")
        cell_mar.append(el)
    tblLook = tblPr.find(qn("w:tblLook"))
    if tblLook is not None:
        tblLook.addprevious(cell_mar)
    else:
        tblPr.append(cell_mar)
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].width = Cm(widths[i])
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(h)
        set_run(r, font, bold=True)
    # 表头跨页重复
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    trPr.append(tblHeader)

    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].width = Cm(widths[i])
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and first_left)
                           else WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            r = p.add_run(str(val))
            set_run(r, font)

    # 最后一行不再与后文绑定；所有行禁止跨页断行，整表随标题走
    for cell in table.rows[-1].cells:
        for p in cell.paragraphs:
            p.paragraph_format.keep_with_next = False
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        row.height = Cm(0.42)
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_three_line(table)

    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.space_after = Pt(0)
    gap.paragraph_format.line_spacing = Pt(4)


def add_figure(name, caption, width=8.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / name), width=Cm(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(0)
    cap.paragraph_format.space_after = Pt(7)
    r = cap.add_run(caption)
    set_run(r, 8)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run(run, 9)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


doc = Document()

# 页面与默认样式
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.0)
sec.right_margin = Cm(2.0)

style = doc.styles["Normal"]
style.font.name = EN_FONT
style.font.size = Pt(10)
style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
style.paragraph_format.space_before = Pt(0)
style.paragraph_format.space_after = Pt(0)

add_page_number(sec)

# 标题区（单栏）
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(4)
r = p.add_run("面向运营容量的电信客户流失预测：")
set_run(r, 16, bold=True, east=CN_HEAD)
r = p.add_run("校准集成与阈值决策")
set_run(r, 16, bold=True, east=CN_HEAD)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("兰丽华")
set_run(r, 11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(8)
r = p.add_run("数据挖掘课程实验报告　　指导教师：高峻嘉")
set_run(r, 9)

# 摘要
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
p.paragraph_format.line_spacing = Pt(12.5)
p.paragraph_format.left_indent = Cm(0.3)
p.paragraph_format.right_indent = Cm(0.3)
r = p.add_run("摘要　")
set_run(r, 9, bold=True, east=CN_HEAD)
abstract = (
    "电信客户流失预测的准确率容易被类别不平衡掩盖，固定使用0.5阈值也缺少业务依据。"
    "本文基于6013条有效客户记录，按探索、清洗、划分、建模、评估、决策六个阶段完成建模，"
    "重点处理了TotalCharges的类型错误与结构性缺失、整行无效记录，以及该字段与在网时长、月费之间的强冗余。"
    "模型保留朴素贝叶斯、支持向量机、多层感知机、决策树和随机森林作为基准，并加入梯度提升树、等渗校准与CatBoost作为对照。"
    "分类阈值在独立验证集上按运营容量确定，测试集只评估一次。结果表明，校准后的梯度提升树取得最高AUC（0.8415）和PR-AUC（0.6556）；"
    "CatBoost直接使用类别字段并未带来提升（AUC为0.8364），说明编码方式不是当前数据的瓶颈。"
    "在每月联系20%客户的约束下，模型可覆盖50.9%的流失客户，提升度为2.54，期望利润约为18.2万美元/千人。"
    "SHAP分析显示，月付合同、短在网时长与缺少技术支持是高风险的主要来源，据此提出合约迁移、新客90天回访和服务包捆绑三类动作。"
)
r = p.add_run(abstract)
set_run(r, 9)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
p.paragraph_format.line_spacing = Pt(12.5)
r = p.add_run("关键词　")
set_run(r, 9, bold=True, east=CN_HEAD)
r = p.add_run("客户流失；类别不平衡；概率校准；CatBoost；SHAP；决策阈值")
set_run(r, 9)

# 正文改为双栏
sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
sec2.top_margin = Cm(2.0)
sec2.bottom_margin = Cm(2.0)
sec2.left_margin = Cm(2.0)
sec2.right_margin = Cm(2.0)
cols = sec2._sectPr.xpath("./w:cols")[0]
cols.set(qn("w:num"), "2")
cols.set(qn("w:space"), "340")

# 1 引言
heading("1　引言")
body(
    "电信行业的获客成本远高于存量维系，客户一旦离网，损失的不只是当月资费，还包括后续生命周期的收入。"
    "因此运营商希望提前识别高风险客户，把有限的挽留预算投向最可能流失、也最值得挽留的人群。"
    "这类问题在数据挖掘课程中常被简化为二分类练习，但真正决定模型价值的往往不是多一个百分点准确率，而是概率是否可信、阈值是否符合运营容量、判断依据能否转化为动作。"
)
body(
    "本文使用课程提供的Telco客户数据，共6014条记录、21个字段，其中流失客户占26.4%。"
    "数据中存在几处容易被忽略的问题：TotalCharges被存成文本并混入空白字符串，末尾有一条只剩客户编号和性别的无效记录，"
    "该字段还与在网时长和月费几乎完全共线。若直接调用dropna或均值填充，这些问题会被掩盖，后续的模型比较也失去意义。"
)
body(
    "围绕任务书要求，本文完成了三项工作。其一，把预处理顺序固定为“先划分、后拟合”，所有补值、编码与缩放都封装在流程内，避免测试信息泄漏。"
    "其二，在课程要求的五个模型之外，加入梯度提升树、概率校准和CatBoost作为对照，检验“让树直接处理类别字段”是否真的更好。"
    "其三，把阈值选择放到独立验证集上，以运营容量和单客经济性为目标，而不是在测试集上反复试阈值。"
)

# 2 数据与探索
heading("2　数据探索与问题定位")
body(
    "样本的流失比例为26.4%，95%置信区间为25.3%至27.5%，正负样本比约为1比2.8。"
    "这意味着全部预测为“不流失”即可得到73.6%的准确率，准确率不适合作为主指标。"
    "本文以PR-AUC和召回率为主，辅以ROC-AUC、Brier分数和对数损失，并为每个指标计算500次Bootstrap置信区间。"
)
body(
    "字段检查发现三处需要处理的问题。第一，TotalCharges的存储类型为object，直接查看缺失值只能发现1条空记录，"
    "但去除首尾空格并强制转换为数值后，有9条空白字符串暴露为缺失，且全部对应在网时长为0的新客户，属于结构性缺失而非随机缺失。"
    "第二，数据末尾有一条仅包含customerID和gender的记录，gender取值为“Fem”，是拼写错误，无法用于监督学习。"
    "第三，TotalCharges与tenure、MonthlyCharges的乘积高度重合，线性回归的R²达到0.9991，说明它基本是可推导的冗余字段。"
)
body(
    "分组统计给出了更直接的业务信号。月付合同的流失率为42.4%，一年合同为11.3%，两年合同只有2.6%；"
    "在网时长不足6个月的新客流失率为52.1%，而48个月以上的老客降到9.2%；"
    "使用光纤接入的客户流失率为41.8%，使用DSL的为18.7%；没有开通技术支持的客户流失率为41.7%，开通后降至14.6%。"
    "这些差异在图1中可以直接观察，也解释了后续SHAP结果为何集中在合同类型、在网时长和服务组合上。"
)
add_figure("fig1_eda.png", "图1　合同类型与在网时长的流失差异")

# 3 方法
heading("3　方法")
heading("3.1　清洗、划分与预处理", size=10.5, before=5)
body(
    "清洗按“记录有效性加领域规则”进行，而不是整表删除缺失。先删除没有标签的无效记录，共1条；"
    "再对TotalCharges去空格并转为数值，把在网时长为0且仍缺失的记录按业务含义补0，"
    "因为新客户尚未产生累计费用；剩余缺失保留为NaN，由流程内的中位数填充处理，并额外生成缺失指示列。"
    "这样既没有编造数据，也保留了“是否缺失”本身可能携带的信息。"
)
body(
    "数据按75比25分层划分为训练集和测试集，再从训练集切出20%作为验证集。验证集只用于选择分类阈值，测试集在全部模型确定后只评估一次。"
    "数值列的补值与标准化、类别列的独热编码都写在ColumnTransformer和Pipeline中，交叉验证时每一折重新拟合，缩放不会提前看到验证数据。"
    "类别编码使用drop first避免完全共线，并设置handle_unknown为ignore，保证测试集出现新类别时不会报错。"
)
heading("3.2　模型设置", size=10.5, before=5)
body(
    "基准模型包括高斯朴素贝叶斯、线性核与RBF核支持向量机、含两个隐层（64和32个单元）的多层感知机、最大深度为4的决策树，以及300棵树的随机森林。"
    "支持向量机使用默认惩罚系数C=1，线性核在4509个训练样本上有2041个支持向量，RBF核有2116个，两者都接近样本的一半。"
    "梯度提升树使用0.05学习率、400轮迭代、31个叶节点，并开启早停；其加权版本使用balanced类别权重，再用5折等渗回归校准输出概率。"
    "CatBoost使用600轮迭代、深度6，直接把合同类型、支付方式等字段作为类别特征输入，不做独热编码。"
)
body(
    "所有模型在训练集上重新拟合后用于测试集评估，阈值仍取自验证集。为判断稳定性，本文对朴素贝叶斯、线性支持向量机、随机森林、梯度提升树、校准模型和CatBoost做了5折交叉验证，"
    "其余模型只报告单次划分结果，报告时会说明这一差异。"
)
heading("3.3　阈值与利润模型", size=10.5, before=5)
body(
    "运营团队每月能触达的客户比例有限，本文取20%作为容量约束，在验证集上取风险分数的80%分位数作为阈值，再原样应用到测试集。"
    "利润模型沿用流失文献的常见设定：成功挽留一名客户带来的收益为24个月月费乘以挽留成功率，挽留成功率用训练集留存率0.7356近似；"
    "每次触达的成本按50美元优惠计算。由此得到每千名客户的期望利润，用于比较不同模型在同一容量下的经济价值。"
)

# 4 结果
heading("4　实验结果")
body(
    "表1汇总了各模型在测试集上的表现。校准后的梯度提升树在AUC和PR-AUC上均排名第一，分别为0.8415和0.6556，5折交叉验证AUC为0.8438，标准差0.0146。"
    "CatBoost的AUC为0.8364，PR-AUC为0.6539，交叉验证均值0.8386，三项都略低于独热编码加梯度提升树的组合，但差距在置信区间之内。"
    "随机森林的AUC为0.8277，线性支持向量机为0.8228，RBF核降到0.7990。朴素贝叶斯的排序能力尚可（AUC为0.8101），但Brier分数高达0.3288，对数损失为2.4407，"
    "说明其输出的概率明显过度自信。多层感知机表现最弱，AUC仅0.7691。"
)
add_table(
    "表1　测试集上的模型指标（阈值来自验证集）",
    ["模型", "AUC", "PR-AUC", "召回", "精确率"],
    [
        ["GBDT+等渗校准", "0.8415", "0.6556", "0.5567", "0.6332"],
        ["CatBoost", "0.8364", "0.6539", "0.4987", "0.6899"],
        ["GBDT", "0.8354", "0.6350", "0.5668", "0.6285"],
        ["随机森林", "0.8277", "0.6269", "0.5542", "0.6377"],
        ["SVM线性核", "0.8228", "0.6246", "0.5567", "0.6022"],
        ["决策树", "0.8118", "0.5744", "0.5239", "0.6082"],
        ["朴素贝叶斯", "0.8101", "0.6020", "0.5264", "0.6076"],
        ["SVM径向基核", "0.7990", "0.6034", "0.5567", "0.6278"],
        ["多层感知机", "0.7691", "0.4967", "0.4912", "0.5342"],
    ],
    widths=[2.7, 1.2, 1.4, 1.15, 1.2],
)
body(
    "图2把AUC与PR-AUC放在同一坐标上并给出置信区间，可以看出排名靠前的几个模型高度重叠。"
    "校准模型与CatBoost的PR-AUC区间分别为0.607至0.705和0.602至0.703，单次划分的差距不足以支撑“谁显著更好”的判断。"
    "图3的ROC与PR曲线也呈现同样的现象，四条曲线在大部分区间几乎重合，差异主要出现在低召回区域。"
)
add_figure("fig2_models.png", "图2　模型排序指标与95%置信区间")
add_figure("fig3_roc_pr.png", "图3　代表性模型的ROC与PR曲线")

body(
    "业务指标换了一个视角。表2显示，线性支持向量机在自身阈值下的利润为20.8万美元/千人，校准模型为19.6万美元；"
    "把容量统一固定为20%后，前者为18.80万美元、后者为18.19万美元，差距约3%。"
    "线性支持向量机的利润略高，但它的概率没有校准，Brier分数为0.1431，交叉验证AUC也低0.0066，二者差距小于Bootstrap区间宽度。"
    "阈值需要迁移到不同月份，预算分配也依赖可信的概率，因此本文选择校准模型作为上线建议。"
    "校准模型在前20%容量下抓到50.9%的流失客户，提升度为2.54。"
)
add_table(
    "表2　20%运营容量下的业务指标（美元/千人）",
    ["模型", "触达率", "召回", "利润@阈值", "利润@20%"],
    [
        ["SVM线性核", "24.4%", "55.7%", "207,960", "188,028"],
        ["GBDT+等渗校准", "23.2%", "55.7%", "196,215", "181,894"],
        ["GBDT", "23.8%", "56.7%", "196,606", "178,841"],
        ["CatBoost", "19.1%", "49.9%", "174,670", "178,822"],
        ["随机森林", "22.9%", "55.4%", "189,545", "172,935"],
        ["朴素贝叶斯", "22.9%", "52.6%", "188,336", "171,700"],
    ],
    widths=[2.4, 1.05, 0.95, 1.6, 1.6],
)
body(
    "CatBoost的结果值得单独说明。它直接使用原始类别字段，省去了独热编码，理论上更适合高基数类别；"
    "但在本数据上并未超过独热编码加梯度提升树的组合，AUC低0.005，PR-AUC低0.002。"
    "原因可能有三点：类别字段的基数都不高（合同类型3类、支付方式4类），独热编码不会造成维度爆炸；"
    "600轮迭代与默认正则化并未针对该数据调参；类别加权后概率整体偏高，阈值被推到0.7614，触达率降到19.1%。"
    "这提醒我们，编码方式的收益依赖数据形态，不能把某个模型的默认优势当成结论。"
)
body(
    "图4给出CatBoost的全局SHAP重要性。合同类型的影响最大，其次是在网时长，然后是网络安全服务、支付方式与接入方式。"
    "高风险组中月付合同占99.7%，全体为55.3%；高风险组在网时长中位数为6个月，全体为29个月；"
    "高风险组有94.1%没有开通技术支持，全体为51.8%。这些差异与第2节的分组统计一致，说明模型学到的是稳定的业务结构，而不是偶然的噪声。"
)
add_figure("fig4_shap.png", "图4　CatBoost的全局SHAP重要性")

body(
    "图5给出利润随触达比例变化的曲线。三条曲线都随容量扩大而上升，但斜率在40%之后明显变缓："
    "高风险人群集中在排序靠前的部分，前几档容量贡献了大部分净收益；继续扩大触达时，新增人群的流失率接近总体水平，"
    "扣除50美元成本后的净增有限。校准模型与GBDT的曲线几乎重合，CatBoost因概率整体偏高、排序略差而位置最低，与表2一致。"
)
add_figure("fig5_profit.png", "图5　利润随运营容量的变化")

# 5 理论分析
heading("5　理论分析")
heading("5.1　朴素贝叶斯的条件独立假设", size=10.5, before=5)
body(
    "朴素贝叶斯把后验概率写成先验与各特征条件概率的连乘，即P(y|x)正比于P(y)乘以所有P(x_j|y)。"
    "这一分解默认在给定类别后各特征相互独立。Telco数据显然不满足该假设：网络安全、在线备份、设备保护、技术支持与接入方式由同一份套餐决定，"
    "条件于流失与否仍然高度相关。独立性不成立时，同一份证据会被重复计入，连乘结果被推向0或1，概率因此过度自信。"
    "实验结果与此吻合：朴素贝叶斯的AUC仍有0.8101，说明排序方向大体正确，但Brier分数为0.3288、对数损失为2.4407，"
    "分别是校准梯度提升树的2.4倍和5.8倍；在验证阈值0.9998下准确率为0.7846，召回率为0.5264。"
    "因此它可以作为快速基线或排序器，但不适合直接输出概率用于利润计算。"
)
heading("5.2　RBF核与线性核的比较", size=10.5, before=5)
body(
    "线性核在原始特征空间寻找一个超平面，RBF核则通过高斯函数把样本映射到无穷维空间，理论上能拟合更复杂的边界。"
    "本文的结果相反：线性核AUC为0.8228，RBF核为0.7990；线性核的支持向量为2041个（占训练集的45.3%），RBF核为2116个（46.9%）。"
    "独热编码后的特征维度约40维，类别边界接近线性，线性核已经能够分开大部分样本；RBF多出的容量没有用于降低偏差，反而拟合了训练集中的局部噪声。"
    "RBF的核宽度由gamma控制，默认值随特征方差变化，在稀疏的独热向量上容易过小，导致每个支持向量的影响范围过窄。"
    "支持向量比例接近也说明RBF没有找到更紧凑的边界。若业务方需要解释系数方向，线性核还具备RBF不具备的可读性。"
)
heading("5.3　多层感知机的损失收敛与XOR问题", size=10.5, before=5)
body(
    "单层感知机只能表示线性决策面，无法解决XOR。本文用四个XOR样本做了对照：逻辑回归的准确率为0.5，"
    "加入一个含4个tanh单元的隐层后，多层感知机达到1.0。隐层的作用是把输入组合成新的中间表示，使原本线性不可分的问题在新空间中可分。"
    "本文的MLP使用64和32个单元，训练402轮后损失从0.5514降到0.1604，曲线后段基本走平，说明在给定学习率与迭代上限下已经收敛。"
    "但测试AUC只有0.7691，是全部模型中最差的。损失收敛只说明优化过程完成，不代表泛化能力；"
    "样本量约4500、输入是稀疏独热向量、网络容量相对任务偏大，都是可能的过拟合来源。"
    "若要做得更稳，应加入L2正则或早停，并用验证损失而不是训练损失选轮次。"
)
heading("5.4　随机森林为何比单棵树稳定", size=10.5, before=5)
body(
    "单棵决策树对训练样本的扰动很敏感：某个分裂点换了一批样本就可能改变，方差因此较大。"
    "Bagging对B棵树取平均，方差可写为ρσ²加(1-ρ)σ²/B，其中ρ是树间相关系数。"
    "随机森林在每次分裂时只考虑随机抽取的特征子集，使树之间的相关性下降，从而压低第二项；"
    "同时每棵树仍是有偏、但方差不会爆炸的模型，平均后方差下降而偏差基本不变。"
    "本文固定测试集，重复抽取10个训练子集后，单棵树的AUC为0.8102±0.0046，随机森林为0.8251±0.0042，"
    "均值提高约0.015，波动也有所下降。降幅不大，是因为特征只有约40维、树已限制深度4，树间相关性仍然偏高。"
    "主实验中随机森林的AUC为0.8277，高于单棵树的0.8118，与上述分析一致。"
)

# 6 业务建议
heading("6　业务取舍与挽留建议")
body(
    "从模型输出到运营动作，还差一步把风险分数翻译成人群和策略。按20%容量计算，校准模型每月可覆盖约五分之一的客户，抓到其中50.9%的流失客户，"
    "精确率为63.3%，即每联系100人约有63人确实会流失。相比随机触达，同样的预算可以多挽回约1.5倍的客户。"
    "若把容量放宽到30%，捕获率升到66.5%，利润由18.2万升到22.5万美元；但继续扩大时增量迅速缩小，从40%到50%只多2.0万，从50%到60%不足0.9万。"
    "考虑到触达人力与客户体验，本文仍把20%作为基准容量，30%可作为预算充足时的备选。"
)
body(
    "高风险画像集中在三类人：月付合同、在网不足一年、没有开通技术支持。对应的动作可以分开设计。"
    "对月付客户，推年付或两年合约并给出首年折扣，直接改变模型中权重最大的风险因素；"
    "对新客，把前90天拆成若干触点，在第一个月内完成套餐确认与使用指导，避免“办完即走”；"
    "对缺少安全与支持服务的客户，用安全加支持的增值包提高转移成本，而不是单纯降价。"
    "这些动作都对应模型中的可干预变量，而不是性别、地区之类无法改变的特征。"
)
body(
    "需要说明的是，SHAP给出的是关联而非因果。模型认为合同类型重要，不代表把客户改成两年合约就一定能降低同样的流失概率。"
    "上线前应做小规模随机对照试验，把人群按风险分层后随机分入干预组与对照组，用实际留存率验证干预效果，再决定是否扩大预算。"
)

# 7 局限与结论
heading("7　局限与结论")
body(
    "本文的结论受限于单次数据划分和单一数据集。虽然使用了Bootstrap区间和5折交叉验证，测试集仍然只有1504条记录，模型排名的置信区间大量重叠。"
    "利润模型中的留存率0.7356与24个月生命周期是近似值，更换行业或优惠成本后，最优容量可能移动。"
    "此外，数据缺少时间维度，无法做跨期验证，概念漂移的风险没有被度量。"
)
body(
    "本文按六个阶段完成了电信客户流失建模，重点不在堆叠模型，而在把预处理顺序、阈值选择和解释环节做扎实。"
    "校准后的梯度提升树在统计指标与业务利润之间取得了较好的平衡；CatBoost的原生类别处理没有带来提升，说明编码方式需要结合数据形态判断；"
    "朴素贝叶斯和RBF核的失败则分别来自条件独立假设与容量过剩。对运营方而言，20%容量、0.478阈值和三类人群策略是可以直接讨论的起点。"
)

# 参考文献
heading("参考文献", size=10.5, before=7)
refs = [
    "[1] Burez J, Van den Poel D. Handling class imbalance in customer churn prediction[J]. Expert Systems with Applications, 2009, 36(3): 4626-4636.",
    "[2] Amin A, et al. Comparing oversampling techniques to handle the class imbalance problem: A customer churn prediction case study[J]. IEEE Access, 2016, 4: 7940-7957.",
    "[3] Prokhorenkova L, Gusev G, Vorobev A, et al. CatBoost: Unbiased boosting with categorical features[C]. NeurIPS, 2018.",
    "[4] Xu T, et al. Telecom churn prediction system based on ensemble learning using feature grouping[J]. Applied Sciences, 2021, 11(11): 5052.",
    "[5] Sikri A, et al. Enhancing customer retention in telecom industry with machine learning driven churn prediction[J]. Scientific Reports, 2024.",
    "[6] Asif M, et al. A data-driven approach with explainable artificial intelligence for customer churn prediction in the telecommunications industry[J]. Results in Engineering, 2025.",
    "[7] The hidden costs of 99% accuracy: A trustworthiness audit of the Telco customer churn benchmark[R]. arXiv:2610.00118, 2026.",
    "[8] e-Profits: A business-aligned evaluation metric for profit-sensitive customer churn prediction[R]. arXiv:2507.08860, 2025.",
    "[9] Lundberg S M, Lee S I. A unified approach to interpreting model predictions[C]. NeurIPS, 2017.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(10.5)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Cm(0.35)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    r = p.add_run(ref)
    set_run(r, 7.5)

# 附录
heading("附录A　小组分工", size=10.5, before=8)
add_table(
    "表3　任务分工（本人独立完成，按任务书角色拆分记录）",
    ["角色", "负责内容", "成员"],
    [
        ["A", "数据探索、缺失诊断与预处理流程", "兰丽华"],
        ["B", "朴素贝叶斯、SVM、MLP及理论分析", "兰丽华"],
        ["C", "基线、树模型、评估与报告、代码走查", "兰丽华"],
    ],
    widths=[1.0, 5.5, 1.5],
)

heading("附录B　代码走查记录", size=10.5, before=6)
add_table(
    "表4　代码走查发现的问题与处理",
    ["问题", "位置", "处理"],
    [
        ["TotalCharges空白串漏检", "数据清洗", "strip后转数值，区分结构性缺失"],
        ["缩放补值早于划分", "预处理", "移入Pipeline，每折重拟合"],
        ["CatBoost克隆报错", "模型封装", "包装器在fit时实例化"],
        ["阈值在测试集上调", "评估", "改用验证集选阈值，只评一次"],
    ],
    widths=[3.3, 1.5, 3.2],
)

heading("附录C　AI使用声明", size=10.5, before=6)
body(
    "本报告在完成过程中使用了AI编程助手，用途限于三类：一是查询scikit-learn与CatBoost的接口用法并定位报错；"
    "二是对绘图和排版代码提供修改建议；三是对文字表达做润色。实验设计、特征处理方案、模型参数选择、结果核验与结论均由本人完成。"
    "文中所有数字来自项目内的churn_pipeline.py及其输出文件，未使用AI生成、篡改或补全实验数据。"
    "AI未参与数据采集，也未代替本人判断模型取舍。若对代码或结果有疑问，可按仓库中的脚本复现全部实验。",
    size=9,
)

doc.core_properties.author = "兰丽华"
doc.core_properties.title = "面向运营容量的电信客户流失预测"
doc.save(OUT_DOCX)
print("saved", OUT_DOCX)
