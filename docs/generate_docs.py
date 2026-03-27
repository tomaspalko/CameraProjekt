"""
Generator PDF dokumentácie pre Weld Inspection Vision System.
Spusti: python docs/generate_docs.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.utils import ImageReader
import datetime
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "WeldInspection_Dokumentacia.pdf")

# ─── Farby ───────────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1a2e4a")
MID_BLUE   = colors.HexColor("#2d5a8e")
LIGHT_BLUE = colors.HexColor("#e8f0fb")
ACCENT     = colors.HexColor("#e85d04")
GREY_LIGHT = colors.HexColor("#f5f5f5")
GREY_MID   = colors.HexColor("#cccccc")
GREY_DARK  = colors.HexColor("#555555")
WHITE      = colors.white
BLACK      = colors.black

# ─── Štýly ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, parent="Normal", **kwargs):
    return ParagraphStyle(name, parent=styles[parent], **kwargs)

style_title     = make_style("DocTitle",    fontSize=28, textColor=WHITE,
                              alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold")
style_subtitle  = make_style("DocSubtitle", fontSize=14, textColor=LIGHT_BLUE,
                              alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica")
style_meta      = make_style("DocMeta",     fontSize=10, textColor=GREY_MID,
                              alignment=TA_CENTER, spaceAfter=2)

style_h1        = make_style("H1",  fontSize=16, textColor=DARK_BLUE, spaceBefore=18,
                              spaceAfter=8, fontName="Helvetica-Bold",
                              borderPadding=(0,0,3,0))
style_h2        = make_style("H2",  fontSize=13, textColor=MID_BLUE, spaceBefore=14,
                              spaceAfter=6, fontName="Helvetica-Bold")
style_h3        = make_style("H3",  fontSize=11, textColor=DARK_BLUE, spaceBefore=10,
                              spaceAfter=4, fontName="Helvetica-Bold")
style_body      = make_style("Body", fontSize=10, textColor=BLACK, spaceAfter=6,
                              leading=15, alignment=TA_JUSTIFY)
style_body_left = make_style("BodyLeft", fontSize=10, textColor=BLACK, spaceAfter=4,
                              leading=14)
style_code      = make_style("Code", fontName="Courier", fontSize=9, textColor=DARK_BLUE,
                              backColor=GREY_LIGHT, spaceAfter=6, spaceBefore=4,
                              leftIndent=12, rightIndent=12, leading=13)
style_note      = make_style("Note", fontSize=9, textColor=GREY_DARK, spaceAfter=4,
                              leftIndent=16, leading=13)
style_bullet    = make_style("Bullet", fontSize=10, textColor=BLACK, spaceAfter=3,
                              leftIndent=16, firstLineIndent=-12, leading=14)
style_toc       = make_style("TOC",  fontSize=11, textColor=DARK_BLUE, spaceAfter=5,
                              leftIndent=8)
style_toc2      = make_style("TOC2", fontSize=10, textColor=GREY_DARK, spaceAfter=3,
                              leftIndent=24)

# ─── Pomocné funkcie ─────────────────────────────────────────────────────────

def h1(text):
    return [HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=4),
            Paragraph(text, style_h1)]

def h2(text):
    return [Paragraph(text, style_h2)]

def h3(text):
    return [Paragraph(text, style_h3)]

def body(text):
    return Paragraph(text, style_body)

def bullet(text):
    return Paragraph(f"• {text}", style_bullet)

def code(text):
    return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), style_code)

def note(text):
    return Paragraph(f"<i>Poznámka: {text}</i>", style_note)

def spacer(h=0.3):
    return Spacer(1, h * cm)

def param_table(rows, col_widths=None):
    """Tabuľka parametrov: [Názov, Typ, Predvolená, Rozsah, Popis]"""
    header = ["Parameter", "Typ", "Predvolená", "Rozsah / Možnosti", "Popis"]
    data = [header] + rows
    if col_widths is None:
        col_widths = [3.5*cm, 1.8*cm, 2.5*cm, 3.8*cm, 5.4*cm]

    ts = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  MID_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  9),
        ("ALIGN",         (0,0), (-1,0),  "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",      (0,1), (-1,-1), 8.5),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GREY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, GREY_MID),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("WORDWRAP",      (0,0), (-1,-1), True),
    ])

    # Wrap cell content
    wrapped = []
    for i, row in enumerate(data):
        if i == 0:
            wrapped.append([Paragraph(f"<b>{c}</b>", make_style(f"th{i}", fontSize=9,
                            textColor=WHITE, fontName="Helvetica-Bold")) for c in row])
        else:
            wrapped.append([Paragraph(str(c), make_style(f"td{i}", fontSize=8.5,
                            leading=12)) for c in row])

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(ts)
    return t

def simple_table(rows, headers, col_widths=None):
    """Jednoduchá tabuľka s hlavičkou."""
    data = [headers] + rows
    if col_widths is None:
        col_widths = None  # auto

    ts = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  DARK_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  9),
        ("FONTSIZE",      (0,1), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GREY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, GREY_MID),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ])

    wrapped = []
    for i, row in enumerate(data):
        st = make_style(f"st{i}", fontSize=9 if i==0 else 8.5,
                        textColor=WHITE if i==0 else BLACK,
                        fontName="Helvetica-Bold" if i==0 else "Helvetica",
                        leading=12)
        wrapped.append([Paragraph(str(c), st) for c in row])

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(ts)
    return t

# ─── Číslovanie strán ─────────────────────────────────────────────────────────

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY_DARK)
    page_num = canvas.getPageNumber()
    canvas.drawRightString(A4[0] - 1.5*cm, 1*cm, f"Strana {page_num}")
    canvas.drawString(1.5*cm, 1*cm, "Weld Inspection Vision System — Dokumentácia")
    canvas.line(1.5*cm, 1.3*cm, A4[0]-1.5*cm, 1.3*cm)
    canvas.restoreState()

def title_page_bg(canvas, doc):
    """Pozadie titulnej strany + číslovanie na ostatných."""
    canvas.saveState()
    if canvas.getPageNumber() == 1:
        # Tmavý header blok
        canvas.setFillColor(DARK_BLUE)
        canvas.rect(0, A4[1]-9*cm, A4[0], 9*cm, fill=1, stroke=0)
        # Accent pruh
        canvas.setFillColor(ACCENT)
        canvas.rect(0, A4[1]-9*cm, A4[0], 0.4*cm, fill=1, stroke=0)
    else:
        add_page_number(canvas, doc)
    canvas.restoreState()

# ─── Obsah dokumentu ─────────────────────────────────────────────────────────

def build_document():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2*cm, bottomMargin=2.2*cm,
        title="Weld Inspection Vision System — Dokumentácia",
        author="WeldVision Team",
        subject="Používateľská a technická dokumentácia",
    )

    story = []

    # ══════════════════════════════════════════════════════════════════
    # TITULNÁ STRANA
    # ══════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 6.5*cm))
    story.append(Paragraph("Weld Inspection Vision System", style_title))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Používateľská a technická dokumentácia", style_subtitle))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(f"Verzia 1.0 &nbsp;|&nbsp; {datetime.date.today().strftime('%d. %m. %Y')}", style_meta))
    story.append(Spacer(1, 8*cm))
    story.append(Paragraph(
        "Systém na vizuálnu inšpekciu zvarových spojov pomocou počítačového videnia. "
        "Určený pre statickú kameru, sub-pixelovú presnosť, riadené osvetlenie.",
        make_style("TitleDesc", fontSize=11, textColor=GREY_DARK,
                   alignment=TA_CENTER, leading=16)
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # OBSAH
    # ══════════════════════════════════════════════════════════════════
    story += h1("Obsah")
    toc_items = [
        ("1", "Prehľad systému"),
        ("2", "Spustenie programu"),
        ("  2.1", "Grafické rozhranie (GUI)"),
        ("  2.2", "Príkazový riadok (CLI)"),
        ("3", "Konfiguračné profily"),
        ("4", "Predspracovanie obrazu"),
        ("5", "Zarovnávacie algoritmy"),
        ("  5.1", "ECC — Enhanced Correlation Coefficient"),
        ("  5.2", "POC — Phase-Only Correlation"),
        ("6", "Detekcia hrán"),
        ("  6.1", "Canny"),
        ("  6.2", "Scharr"),
        ("  6.3", "LoG — Laplacian of Gaussian"),
        ("  6.4", "PhaseCongruency"),
        ("  6.5", "DexiNed (hlboké učenie)"),
        ("7", "ROI — Oblasť záujmu"),
        ("8", "Kalibrácia (px → mm)"),
        ("9", "GUI — Inšpekčný panel"),
        ("10", "GUI — Batch panel"),
        ("11", "Výstupné formáty"),
        ("12", "Technické limity a tolerancie"),
    ]
    for num, title in toc_items:
        indent = 8 if num.startswith("  ") else 0
        st = make_style(f"toc_{num}", fontSize=10 if indent else 11,
                        textColor=GREY_DARK if indent else DARK_BLUE,
                        leftIndent=indent+8, spaceAfter=4,
                        fontName="Helvetica" if indent else "Helvetica-Bold")
        story.append(Paragraph(f"{num.strip()}. &nbsp; {title}", st))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 1. PREHĽAD SYSTÉMU
    # ══════════════════════════════════════════════════════════════════
    story += h1("1. Prehľad systému")
    story.append(body(
        "Weld Inspection Vision System je desktopová aplikácia na automatizovanú vizuálnu kontrolu "
        "zvarových spojov na kovových dielcoch. Systém porovnáva referenčný obraz s inšpekčným "
        "obrazom a meria posunutie (dx, dy) a rotáciu objektu s sub-pixelovou presnosťou."
    ))
    story.append(spacer(0.3))

    story += h2("Technické parametre")
    spec_rows = [
        ["Kamera",         "Statická, do 2 MP (1280×1024)"],
        ["Objekt",         "Kovový zvarový spoj"],
        ["Presnosť",       "Sub-pixelová (cieľ &lt; 0,05 px, 1/100 px)"],
        ["Pohyb",          "Malý posun, rotácia ~1°"],
        ["Osvetlenie",     "Riadené, stabilné"],
        ["Výstup",         "dx, dy, uhol (°), spoľahlivosť, dx_mm, dy_mm"],
        ["Režim",          "GUI aj batch (uložené obrazy)"],
        ["Rozhranie",      "Desktop GUI (PyQt6)"],
    ]
    story.append(simple_table(spec_rows, ["Paramter", "Hodnota"], [5*cm, 12*cm]))
    story.append(spacer(0.5))

    story += h2("Algoritmus pipeline")
    story.append(body("Spracovanie prebieha v nasledovných krokoch:"))
    steps = [
        "Načítanie obrazu (PNG, JPG, BMP, TIFF)",
        "Predspracovanie: konverzia na odtiene sivej → CLAHE (kontrastné vyrovnanie) → Gaussovo rozmazanie",
        "Aplikácia ROI masky (ak je definovaná)",
        "Zarovnanie: ECC alebo POC algoritmus → dx_px, dy_px, uhol_deg, spoľahlivosť",
        "Konverzia: dx_px × mm_per_px → dx_mm, dy_mm",
        "Výstup výsledkov (GUI overlay, CSV, JSON)",
    ]
    for i, s in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{i}.</b> {s}", style_bullet))
    story.append(spacer(0.3))

    story += h2("Súradnicový systém")
    story.append(body(
        "Systém používa OpenCV konvenciu: <b>dx &gt; 0</b> = posun doprava, "
        "<b>dy &gt; 0</b> = posun nadol. Uhol je kladný pre rotáciu v smere hodinových ručičiek."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 2. SPUSTENIE
    # ══════════════════════════════════════════════════════════════════
    story += h1("2. Spustenie programu")

    story += h2("2.1 Grafické rozhranie (GUI)")
    story.append(body("Spustenie GUI aplikácie:"))
    story.append(code("python main.py --gui"))
    story.append(body(
        "GUI obsahuje dve záložky: <b>Inšpekcia</b> (nastavenia, test zarovnania, overlay) "
        "a <b>Batch</b> (hromadné spracovanie)."
    ))
    story.append(spacer(0.5))

    story += h2("2.2 Príkazový riadok (CLI)")
    story.append(body("CLI umožňuje batch spracovanie bez grafického rozhrania:"))
    story.append(code(
        "# Batch bez profilu<br/>"
        "python main.py --reference data/reference/ref.png --folder data/batch/ --csv results/out.csv<br/><br/>"
        "# Batch s profilom<br/>"
        "python main.py --profile job_01 --folder data/batch/ --json results/out.json --verbose"
    ))
    story.append(spacer(0.3))

    cli_rows = [
        ["--gui",             "príznak",  "—",                    "Spustí GUI (ignoruje ostatné argumenty)"],
        ["--reference PATH",  "reťazec",  "povinný*",             "Cesta k referenčnému obrazu"],
        ["--folder PATH",     "reťazec",  "povinný",              "Priečinok so zdrojovými obrazmi"],
        ["--profile NAME",    "reťazec",  "—",                    "Názov uloženého profilu (nahrádza --reference)"],
        ["--profiles-dir DIR","reťazec",  "config/profiles",      "Priečinok s profilmi"],
        ["--csv PATH",        "reťazec",  "—",                    "Cesta na uloženie výsledkov vo formáte CSV"],
        ["--json PATH",       "reťazec",  "—",                    "Cesta na uloženie výsledkov vo formáte JSON (so štatistikami)"],
        ["--verbose / -v",    "príznak",  "—",                    "Vypíše výsledky každého obrazu do konzoly"],
    ]
    story.append(simple_table(cli_rows, ["Argument", "Typ", "Predvolená", "Popis"],
                               [4*cm, 2.2*cm, 3*cm, 7.8*cm]))
    story.append(note("*Buď --profile alebo --reference musí byť zadaný."))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 3. KONFIGURAČNÉ PROFILY
    # ══════════════════════════════════════════════════════════════════
    story += h1("3. Konfiguračné profily")
    story.append(body(
        "Profil je súbor všetkých nastavení konkrétnej inšpekčnej úlohy. "
        "Ukladá sa ako JSON súbor v priečinku <b>config/profiles/&lt;názov&gt;.json</b>. "
        "Profil umožňuje rýchle prepínanie medzi rôznymi typmi dielcov alebo inšpekčnými podmienkami."
    ))
    story.append(spacer(0.3))

    story += h2("Polia profilu")
    prof_rows = [
        ["name",             "reťazec", "—",         "neprázdny",         "Jedinečný názov profilu (použitý ako názov súboru)"],
        ["reference_path",   "reťazec", "\"\"",       "platná cesta",      "Cesta k referenčnému obrazu"],
        ["scale_mm_per_px",  "float",   "1.0",        "> 0",               "Fyzická mierka pre konverziu px na mm"],
        ["roi",              "objekt",  "None",       "x0&lt;x1, y0&lt;y1", "Obdĺžnik záujmu (x0,y0,x1,y1). None = celý obraz"],
        ["algorithm",        "reťazec", "\"ECC\"",    "ECC, POC",          "Zarovnávací algoritmus"],
        ["ecc_max_iter",     "int",     "2000",       ">= 1",              "Max. počet iterácií ECC"],
        ["ecc_epsilon",      "float",   "1e-8",       "> 0",               "Prahová hodnota konvergencie ECC"],
        ["ecc_gauss_filt_size","int",   "3",          "1, 3, 5, 7",        "Veľkosť Gaussovho filtra pre gradienty ECC"],
        ["auto_clahe",       "bool",    "False",      "True / False",      "Adaptívne nastavenie clip limitu CLAHE podľa gradientov"],
        ["edge_method",      "reťazec", "\"Canny\"",  "Canny/Scharr/LoG/PhaseCongruency/DexiNed", "Metóda detekcie hrán pre overlay"],
        ["canny_threshold1", "int",     "50",         "0–255",             "Dolný prah Canny hysterézy"],
        ["canny_threshold2", "int",     "150",        "0–255",             "Horný prah Canny hysterézy"],
        ["canny_blur",       "int",     "3",          "1,3,5,7… (nepárne)","Veľkosť Gaussovho rozmazania pred Canny (1=vypnuté)"],
        ["scharr_threshold", "int",     "30",         "0–255",             "Prah magnitudácie gradientu Scharr"],
        ["scharr_blur",      "int",     "3",          "1,3,5,7… (nepárne)","Gaussovo rozmazanie pred Scharr"],
        ["log_sigma",        "float",   "1.5",        "> 0",               "Smerodajná odchýlka Gaussovho jadra pre LoG"],
        ["log_threshold",    "int",     "10",         "0–255",             "Prah LoG odpovede"],
        ["log_blur",         "int",     "3",          "1,3,5,7… (nepárne)","Predbehu rozmazanie pre LoG"],
        ["pc_nscale",        "int",     "4",          "1–8",               "Počet mierkok filtra PhaseCongruency"],
        ["pc_min_wavelength","int",     "6",          "2–40",              "Najmenšia vlnová dĺžka filtra [px]"],
        ["pc_mult",          "float",   "2.1",        "> 1.0",             "Faktor mierky medzi susednými filtrami"],
        ["pc_k",             "float",   "2.0",        "> 0",               "Multiplikátor prahu šumu"],
        ["dexined_weights",  "reťazec", "\"\"",       "cesta k .onnx",     "Cesta k váham DexiNed modelu (prázdne = predvolené)"],
        ["dexined_threshold","float",   "0.5",        "0.0–1.0",           "Prah sigmoid pre binarizáciu mapy hrán"],
        ["dexined_device",   "reťazec", "\"cpu\"",    "cpu, cuda",         "Výpočtové zariadenie (cuda = NVIDIA GPU)"],
    ]
    story.append(param_table(prof_rows))
    story.append(spacer(0.4))

    story += h2("Príklad JSON profilu")
    story.append(code(
        '{\n'
        '  "name": "job_01",\n'
        '  "reference_path": "data/reference/weld.png",\n'
        '  "roi": {"x0": 100, "y0": 50, "x1": 500, "y1": 450},\n'
        '  "scale_mm_per_px": 0.05,\n'
        '  "algorithm": "ECC",\n'
        '  "ecc_max_iter": 2000,\n'
        '  "ecc_epsilon": 1e-8,\n'
        '  "auto_clahe": false,\n'
        '  "edge_method": "Canny",\n'
        '  "canny_threshold1": 50,\n'
        '  "canny_threshold2": 150\n'
        '}'
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 4. PREDSPRACOVANIE
    # ══════════════════════════════════════════════════════════════════
    story += h1("4. Predspracovanie obrazu")
    story.append(body(
        "Predspracovanie pripraví vstupný obraz pre zarovnávacie algoritmy. "
        "Prebieha automaticky pri každom zarovnaní a pozostáva z troch krokov: "
        "konverzia na odtiene sivej, CLAHE kontrastné vyrovnanie a Gaussovo rozmazanie."
    ))
    story.append(spacer(0.3))

    story += h2("Parametre predspracovania")
    pre_rows = [
        ["clahe_clip",   "float", "2.0",  "> 0; typicky 1–4",     "Clip limit CLAHE — sila kontrastného vyrovnania. Vyššia hodnota = silnejší kontrast, ale môže zosíliť šum."],
        ["blur_kernel",  "int",   "5",    "nepárne > 0; typicky 3, 5, 7", "Veľkosť Gaussovho jadra rozmazania. Väčší kernel = silnejšie vyhladenie šumu, ale strata detailov hrán."],
        ["auto_clahe",   "bool",  "False","True / False",          "Ak True: clip limit sa vypočíta automaticky z gradientov obrazu: clip = clamp(2.0 × 50 / std_gradientu, 1.0, 4.0). Ignoruje clahe_clip."],
    ]
    story.append(param_table(pre_rows))
    story.append(spacer(0.3))

    story += h2("CLAHE — Contrast Limited Adaptive Histogram Equalization")
    story.append(body(
        "CLAHE vyrovnáva histogram adaptívne po malých oblastiach obrazu (tiles), čím zvyšuje "
        "lokálny kontrast bez nadmerného zosilnenia šumu. Je obzvlášť účinný pri nerovnomernom "
        "osvetlení. Clip limit obmedzuje maximálny gradient histogramu — chrání pred artefaktmi."
    ))

    story += h2("Automatický CLAHE (auto_clahe=True)")
    story.append(body(
        "Keď je zapnutý auto_clahe, systém zmeria rozptyl Sobelovho gradientu referenčného obrazu "
        "a nastaví clip limit dynamicky. Obraz s malým kontrastom dostane vyšší clip (silnejšie "
        "vyrovnanie), obraz s vysokým kontrastom dostane nižší clip (jemnejšia úprava)."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 5. ZAROVNÁVACIE ALGORITMY
    # ══════════════════════════════════════════════════════════════════
    story += h1("5. Zarovnávacie algoritmy")
    story.append(body(
        "Systém podporuje dva zarovnávacie algoritmy: ECC (primárny) a POC (záložný). "
        "Oba algoritmy predpokladajú malý pohyb: posun do ~15 px, rotácia do ~2°."
    ))
    story.append(spacer(0.3))

    story += h2("5.1 ECC — Enhanced Correlation Coefficient")
    story.append(body(
        "ECC je iteratívny algoritmus implementovaný v OpenCV (<b>cv2.findTransformECC</b>). "
        "Hľadá euklidovskú transformáciu (posun + rotácia) ktorá maximalizuje koreláciu "
        "medzi referenčným a inšpekčným obrazom. Inicializuje sa z matice identity — "
        "pre ~1° rotáciu to postačuje bez hrubého odhadu polohy."
    ))
    story.append(spacer(0.3))

    ecc_rows = [
        ["max_iter",        "int",   "2000",  ">= 100; typicky 1000–5000", "Maximálny počet iterácií. Viac iterácií = vyššia presnosť, ale dlhší čas výpočtu."],
        ["epsilon",         "float", "1e-8",  "> 0; typicky 1e-6 až 1e-10","Prahová hodnota konvergencie. Keď zmena klesne pod epsilon, algoritmus sa zastaví."],
        ["gauss_filt_size", "int",   "3",     "1, 3, 5, 7 (nepárne)",      "Veľkosť Gaussovho filtra pre výpočet gradientov. 1 = bez filtra. Väčší kernel tlmí šum, ale môže znižovať presnosť."],
    ]
    story.append(param_table(ecc_rows))
    story.append(spacer(0.3))

    story += h2("ECC — technické poznámky")
    story.append(bullet(
        "<b>Oprava presnosti uhla:</b> Systém používa asin(warp[1,0]) namiesto acos(warp[0,0]) "
        "pre výpočet uhla. Funkcia acos má zaokrúhľovaciu chybu blízko 0°; asin dáva lepšiu "
        "presnosť pri malých rotáciách."
    ))
    story.append(bullet(
        "<b>Pyramída = 1:</b> Pri malom pohybe multi-scale prístup neprináša výhody. "
        "Systém používa jedinú mierku pre maximálnu presnosť."
    ))
    story.append(bullet(
        "<b>Spoľahlivosť:</b> ECC vracia korelačnú hodnotu [0, 1]. Odporúčaný minimálny "
        "prah: 0.7. Hodnoty pod 0.6 naznačujú zlyhaný prípad."
    ))
    story.append(spacer(0.4))

    story += h2("5.2 POC — Phase-Only Correlation")
    story.append(body(
        "POC je záložný algoritmus využívajúci frekvenčnú doménu (FFT). "
        "Pozostáva z dvoch krokov: (1) log-polárna FFT pre odhad rotácie, "
        "(2) klasická POC s Hannovým oknom pre sub-pixelový odhad posunu. "
        "Je vhodný pre prípadky kde ECC konverguje pomaly alebo zlyháva."
    ))
    story.append(spacer(0.3))

    poc_rows = [
        ["reference", "ndarray", "—",    "uint8 grayscale", "Referenčný obraz"],
        ["image",     "ndarray", "—",    "uint8 grayscale", "Inšpekčný obraz na zarovnanie"],
        ["mask",      "ndarray", "None", "uint8 / None",    "Voliteľná ROI maska (aplikuje sa len pri odhade posunu, nie rotácie)"],
    ]
    story.append(param_table(poc_rows))
    story.append(spacer(0.3))
    story.append(note(
        "POC používa pevné interné parametre: log-polárne rozlíšenie 512×log_radius, "
        "rozsah rotácie -90° až +90°, parabolický fit sub-pixelového maxima."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 6. DETEKCIA HRÁN
    # ══════════════════════════════════════════════════════════════════
    story += h1("6. Detekcia hrán")
    story.append(body(
        "Detekcia hrán sa používa pre vizualizačný overlay v GUI — zobrazuje kontúry "
        "objektu na inšpekčnom obraze. Systém podporuje päť metód s rôznymi "
        "charakteristikami rýchlosti a robustnosti."
    ))
    story.append(spacer(0.2))

    methods_summary = [
        ["Canny",           "Rýchla", "Dobrá pre ostré hrany s hysterézou"],
        ["Scharr",          "Rýchla", "Citlivá na smer hrán, odolnejšia pri šume"],
        ["LoG",             "Stredná","Detekuje prechody nulou, vhodná pre hladký šum"],
        ["PhaseCongruency", "Pomalá", "Invariantná voči kontrastu, ideálna pre priemyselné povrchy"],
        ["DexiNed",         "Pomalá", "Hlboké učenie, najvyššia presnosť, vyžaduje GPU pre rýchlosť"],
    ]
    story.append(simple_table(methods_summary, ["Metóda", "Rýchlosť", "Charakteristika"],
                               [4.5*cm, 3*cm, 9.5*cm]))
    story.append(spacer(0.4))

    story += h2("6.1 Canny")
    story.append(body(
        "Canny je štandardná metóda detekcie hrán s hysterézou. Najprv rozmaže obraz "
        "Gaussovým filtrom, vypočíta gradienty a aplikuje ne-maximálnu potlač. "
        "Hysteréza s dvoma prahmi zabezpečuje spojitosť hrán."
    ))
    canny_rows = [
        ["threshold1", "int", "50",  "0–255",              "Dolný prah hysterézy. Hrany so silou &lt; t1 sú zamietnuté."],
        ["threshold2", "int", "150", "0–255",              "Horný prah hysterézy. Hrany so silou > t2 sú prijaté. Hrany medzi t1 a t2 sú prijaté len ak susedia s prijatou hranou."],
        ["blur",       "int", "3",   "1,3,5,7… (nepárne)","Gaussovo rozmazanie pred Canny. 1 = bez rozmazania. Vyššie hodnoty redukujú šum."],
    ]
    story.append(param_table(canny_rows))
    story.append(note("Odporúčaný pomer: threshold2 ≈ 2–3 × threshold1."))
    story.append(spacer(0.4))

    story += h2("6.2 Scharr")
    story.append(body(
        "Scharr vypočíta magnitudáciu gradientu pomocou optimalizovaného Scharrových operátorov "
        "(presnejší ako Sobel pri malom jadre). Výsledok je normalizovaný na 0–255 a prahovaný."
    ))
    scharr_rows = [
        ["threshold", "int", "30", "0–255",              "Prah pre magnitudáciu Scharr gradientu (normalizovaný na 0–255)."],
        ["blur",      "int", "3",  "1,3,5,7… (nepárne)","Gaussovo rozmazanie pred výpočtom. 1 = bez rozmazania."],
    ]
    story.append(param_table(scharr_rows))
    story.append(spacer(0.4))

    story += h2("6.3 LoG — Laplacian of Gaussian")
    story.append(body(
        "LoG konvolvuje obraz s Laplaciánom Gaussovej funkcie a detekuje prechody nulou "
        "(zero-crossings). Je odolnejší voči šumu ako prostý Laplacián. "
        "Sigma určuje mierku hrán — väčšia sigma detekuje hrubšie štruktúry."
    ))
    log_rows = [
        ["sigma",     "float", "1.5", "> 0; typicky 0.5–3.0",  "Smerodajná odchýlka Gaussovho jadra. Väčšia sigma = hrubšie/väčšie hrany, menšia = jemné hrany."],
        ["threshold", "int",   "10",  "0–255",                 "Prah LoG odpovede (normalizovaný). Oddeľuje skutočné hrany od šumu."],
        ["blur",      "int",   "3",   "1,3,5,7… (nepárne)",   "Predbehu Gaussovo rozmazanie. 1 = bez rozmazania."],
    ]
    story.append(param_table(log_rows))
    story.append(spacer(0.4))

    story += h2("6.4 PhaseCongruency")
    story.append(body(
        "PhaseCongruency detekuje hrany ako miesta kde frekvenčné zložky obrazu majú "
        "maximálnu zhodu vo fáze. Táto vlastnosť ju robí invariantnou voči zmene osvetlenia "
        "a kontrastu — ideálna pre kovové povrchy s odrazmi."
    ))
    pc_rows = [
        ["nscale",         "int",   "4",   "1–8",              "Počet mierkok filtra. Viac mierkok = väčšia robustnosť, ale pomalší výpočet."],
        ["min_wavelength", "int",   "6",   "2–40 [px]",        "Najmenšia vlnová dĺžka Gaborovho filtra v pixeloch. Určuje najjemnejšie hrany ktoré sú detekovateľné."],
        ["mult",           "float", "2.1", "> 1.0",            "Faktor medzi susednými mierkami. Väčší faktor = väčší rozsah detekovaných mieriek."],
        ["k",              "float", "2.0", "> 0; typicky 1–5", "Násobiteľ prahu šumu. Vyššia hodnota = prísnejší prah, menej falošných hrán."],
    ]
    story.append(param_table(pc_rows))
    story.append(spacer(0.4))

    story += h2("6.5 DexiNed (hlboké učenie)")
    story.append(body(
        "DexiNed (Deep Extreme Image Detection) je konvolučná neurónová sieť trénovaná "
        "na detekciu hrán. Beží cez ONNX runtime — kompatibilná s CPU aj NVIDIA GPU. "
        "Produkuje pravdepodobnostnú mapu hrán, ktorá je binarizovaná prahom."
    ))
    dex_rows = [
        ["weights_path", "reťazec", "\"\"",     "cesta k .onnx súboru", "Cesta k váhovému súboru modelu. Prázdne = použiť predvolené (models/dexined.onnx)."],
        ["threshold",    "float",   "0.5",      "0.0–1.0",              "Sigmoid prah pre binarizáciu mapy hrán. Nižší = viac hrán (aj šum), vyšší = menej hrán."],
        ["device",       "reťazec", "\"cpu\"",  "cpu, cuda",            "Výpočtové zariadenie. cuda vyžaduje NVIDIA GPU s CUDA drivermi nainštalovanými."],
    ]
    story.append(param_table(dex_rows))
    story.append(note(
        "DexiNed je najpomalšia metóda pri inferencii na CPU. Pre real-time použitie "
        "odporúčame GPU alebo rýchlejšiu metódu (Canny, Scharr)."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 7. ROI
    # ══════════════════════════════════════════════════════════════════
    story += h1("7. ROI — Oblasť záujmu (Region of Interest)")
    story.append(body(
        "ROI definuje obdĺžnikovú oblasť v referenčnom obraze na ktorú sa sústreďuje "
        "zarovnávací algoritmus. Ignorovanie okrajov obrazu (kde môže byť pozadie, "
        "upnutie dielca, odlesky) zvyšuje presnosť a rýchlosť zarovnania."
    ))
    story.append(spacer(0.3))

    roi_rows = [
        ["x0", "int", "—", "&gt;= 0, &lt; x1",   "X-koordinát ľavého horného rohu (inkluzívny)"],
        ["y0", "int", "—", "&gt;= 0, &lt; y1",   "Y-koordinát ľavého horného rohu (inkluzívny)"],
        ["x1", "int", "—", "> x0",          "X-koordinát pravého dolného rohu (exkluzívny — numpy konvencia)"],
        ["y1", "int", "—", "> y0",          "Y-koordinát pravého dolného rohu (exkluzívny)"],
    ]
    story.append(param_table(roi_rows, [3*cm, 1.8*cm, 2*cm, 3.5*cm, 6.7*cm]))
    story.append(spacer(0.3))

    story += h2("Definovanie ROI v GUI")
    story.append(bullet("Kliknúť na <b>\"Kresliť ROI\"</b> — kurzor sa zmení na krížik"))
    story.append(bullet("Potiahnuť myšou cez referenčný obraz — oranžový obdĺžnik ukazuje výber"))
    story.append(bullet("Uvoľniť tlačidlo myši — ROI sa uloží, koordináty sa zobrazia v políčkach"))
    story.append(bullet("Tlačidlo <b>\"Zmazať ROI\"</b> odstráni obmedenie (použije sa celý obraz)"))
    story.append(bullet("Koordináty je možné zadať aj priamo do políčok x0, y0, x1, y1"))
    story.append(spacer(0.3))
    story.append(note(
        "Koordináty používajú numpy konvenciu: x1 a y1 sú exkluzívne. "
        "Napríklad ROI (100, 50, 500, 450) má šírku 400 px a výšku 400 px."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 8. KALIBRÁCIA
    # ══════════════════════════════════════════════════════════════════
    story += h1("8. Kalibrácia (px → mm)")
    story.append(body(
        "Kalibrácia definuje fyzickú mierku medzi pixelmi a milimetrami. "
        "Umožňuje vyjadriť namerané posunutia v reálnych fyzikálnych jednotkách "
        "pre protokoly a záznamy kvality."
    ))
    story.append(spacer(0.3))

    cal_rows = [
        ["scale_mm_per_px", "float", "1.0", "> 0", "Fyzická mierka: počet milimetrov zodpovedajúci jednému pixelu. Konverzia: dx_mm = dx_px × scale_mm_per_px"],
    ]
    story.append(param_table(cal_rows, [4*cm, 1.8*cm, 2.5*cm, 2*cm, 6.7*cm]))
    story.append(spacer(0.3))

    story += h2("Určenie kalibrovanej hodnoty")
    story.append(body("Postup kalibrácie pomocou referenčného objektu:"))
    steps_cal = [
        "Položiť objekt so známou fyzickou dĺžkou do zorného poľa kamery",
        "Fotografovať objekt za rovnakých podmienok ako inšpekčné obrazy",
        "Zmerať dĺžku objektu v pixeloch v softvéri (napr. GIMP, ImageJ)",
        "Vypočítať: scale = fyzická_dĺžka_mm / dĺžka_px",
        "Zadať výslednú hodnotu do políčka Kalibrácia (mm/px) v GUI",
    ]
    for i, s in enumerate(steps_cal, 1):
        story.append(Paragraph(f"<b>{i}.</b> {s}", style_bullet))

    story.append(spacer(0.3))
    story.append(note(
        "Ak kalibrácia nie je potrebná (výstup len v pixeloch), ponechajte predvolenú hodnotu 1.0. "
        "Výsledky dx_mm a dy_mm budú numericky rovnaké ako dx_px, dy_px."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 9. GUI — INŠPEKČNÝ PANEL
    # ══════════════════════════════════════════════════════════════════
    story += h1("9. GUI — Inšpekčný panel")
    story.append(body(
        "Inšpekčný panel (záložka <b>Inšpekcia</b>) je hlavné pracovné prostredie. "
        "Obsahuje viewer referenčného obrazu s ROI editáciou a panel nastavení vpravo."
    ))
    story.append(spacer(0.3))

    gui_groups = [
        ("Referenčný obraz", [
            ("Načítaj obraz…", "Tlačidlo", "Otvorí dialóg na výber referenčného obrazového súboru (PNG, JPG, BMP, TIFF)"),
        ]),
        ("ROI", [
            ("Kresliť ROI", "Prepínač", "Zapína/vypína režim kreslenia ROI myšou (rubber-band selection)"),
            ("Zmazať ROI",  "Tlačidlo", "Odstráni definovanú ROI (zarovnanie bude prebiehať na celom obraze)"),
            ("x0, y0",      "SpinBox int", "Ľavý horný roh ROI v pixeloch (0–9999)"),
            ("x1, y1",      "SpinBox int", "Pravý dolný roh ROI v pixeloch (0–9999)"),
        ]),
        ("Kalibrácia", [
            ("Scale (mm/px)", "SpinBox float", "Fyzická mierka. Rozsah: 0.0001–100.0, 6 desatinných miest"),
        ]),
        ("Algoritmus", [
            ("Algoritmus",          "Combo",  "Výber: ECC alebo POC"),
            ("Max iterations",      "SpinBox int", "Maximálny počet iterácií ECC (100–10000)"),
            ("Epsilon",             "TextField", "Konvergenčný prah ECC (napr. 1e-08)"),
            ("Gauss filter size",   "Combo int", "Veľkosť Gaussovho filtra pre gradienty: 1, 3, 5, 7"),
            ("Auto CLAHE",          "CheckBox", "Zapína adaptívne nastavenie CLAHE clip limitu z gradientov"),
        ]),
        ("Detekcia hrán", [
            ("Edge method",         "Combo",  "Metóda detekcie hrán pre overlay: Canny, Scharr, LoG, PhaseCongruency, DexiNed"),
            ("Threshold 1 / 2",     "SpinBox int", "Prahové hodnoty Canny hysterézy (0–255)"),
            ("Blur kernel",         "Combo int", "Gaussovo rozmazanie pred detekciou: 1, 3, 5, 7, 9, 11"),
            ("Scharr Threshold",    "SpinBox int", "Prah Scharr magnitudácie (0–255)"),
            ("LoG Sigma",           "SpinBox float", "Sigma LoG filtra (0.1–10.0)"),
            ("LoG Threshold",       "SpinBox int", "Prah LoG odpovede (0–255)"),
            ("N-scales",            "SpinBox int", "Počet mierkok PhaseCongruency (1–8)"),
            ("Min wavelength",      "SpinBox int", "Najmenšia vlnová dĺžka PhaseCongruency [px] (2–40)"),
            ("Multiplier",          "SpinBox float", "Faktor mierky PhaseCongruency (1.0–3.0)"),
            ("k (noise)",           "SpinBox float", "Prah šumu PhaseCongruency (0.5–5.0)"),
            ("DexiNed weights",     "TextField", "Cesta k .onnx modelu (prázdne = predvolené)"),
            ("DexiNed threshold",   "SpinBox float", "Sigmoid prah DexiNed (0.0–1.0)"),
            ("Device",              "Combo", "Výpočtové zariadenie DexiNed: cpu, cuda"),
        ]),
        ("Profil", [
            ("Názov profilu",  "TextField", "Názov pod ktorým sa uloží konfigurácia"),
            ("Uložiť profil",  "Tlačidlo",  "Uloží aktuálne nastavenia ako JSON profil"),
            ("Načítať profil", "Combo + Tlačidlo", "Vyberá a načítava uložený profil"),
            ("Zmazať profil",  "Tlačidlo",  "Zmaže vybraný profil zo súborového systému"),
        ]),
        ("Test zarovnania", [
            ("Načítaj testovací obraz…", "Tlačidlo", "Výber inšpekčného obrazu na otestovanie zarovnania"),
            ("Spustiť zarovnanie",       "Tlačidlo", "Spustí zarovnanie a zobrazí výsledky + overlay"),
            ("Výsledky",                 "Textový panel", "Zobrazuje: dx_px, dy_px, dx_mm, dy_mm, uhol°, spoľahlivosť, NCC skóre"),
        ]),
    ]

    for group_name, controls in gui_groups:
        story += h2(group_name)
        ctrl_rows = [[c[0], c[1], c[2]] for c in controls]
        story.append(simple_table(ctrl_rows, ["Ovládací prvok", "Typ", "Popis"],
                                   [4.5*cm, 3.5*cm, 9*cm]))
        story.append(spacer(0.3))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 10. BATCH PANEL
    # ══════════════════════════════════════════════════════════════════
    story += h1("10. GUI — Batch panel")
    story.append(body(
        "Batch panel (záložka <b>Batch</b>) umožňuje hromadné spracovanie priečinka "
        "s obrazmi. Spracovanie beží v samostatnom vlákne — GUI zostáva responzívne "
        "a tabuľka výsledkov sa aktualizuje za behu."
    ))
    story.append(spacer(0.3))

    story += h2("Nastavenia batch spracovania")
    batch_ctrl = [
        ["Priečinok",      "TextField + Prehľadávať", "Cesta k priečinku so zdrojovými obrazmi"],
        ["Profil",         "Combo",                    "Výber uloženého profilu (obsahuje všetky nastavenia vrátane ref. obrazu)"],
        ["Obnoviť",        "Tlačidlo",                 "Znovu načíta zoznam dostupných profilov"],
        ["Export CSV",     "TextField + Prehľadávať", "Cesta k výstupnému CSV súboru s výsledkami"],
        ["Export JSON",    "TextField + Prehľadávať", "Cesta k výstupnému JSON súboru (obsahuje aj štatistiky)"],
        ["Spustiť batch",  "Tlačidlo",                 "Spustí spracovanie všetkých obrazov v priečinku"],
        ["Progress bar",   "Indikátor",                "Zobrazuje priebeh spracovania (0–100%)"],
    ]
    story.append(simple_table(batch_ctrl, ["Ovládací prvok", "Typ", "Popis"],
                               [4*cm, 4.5*cm, 8.5*cm]))
    story.append(spacer(0.4))

    story += h2("Tabuľka výsledkov")
    story.append(body("Každý spracovaný obraz sa zobrazí ako riadok tabuľky za behu:"))
    cols = [
        ["Súbor",        "Názov súboru obrazu"],
        ["Stav",         "OK (úspešné) alebo ERROR (chyba pri spracovaní)"],
        ["dx_px",        "Horizontálny posun v pixeloch"],
        ["dy_px",        "Vertikálny posun v pixeloch"],
        ["dx_mm",        "Horizontálny posun v milimetroch"],
        ["dy_mm",        "Vertikálny posun v milimetroch"],
        ["Uhol°",        "Rotácia v stupňoch"],
        ["Spoľahlivosť", "Korelačná hodnota zarovnania [0, 1]"],
    ]
    story.append(simple_table(cols, ["Stĺpec", "Popis"], [4*cm, 13*cm]))
    story.append(spacer(0.4))

    story += h2("Štatistiky")
    story.append(body(
        "Po dokončení batch spracovania sa zobrazí súhrn štatistík pre všetky metriky:"
    ))
    stat_cols = [
        ["Počet",       "Celkový počet / OK / chybné"],
        ["Priemer ± Std", "Stredná hodnota a smerodajná odchýlka"],
        ["Min / Max",   "Minimálna a maximálna nameraná hodnota"],
    ]
    story.append(simple_table(stat_cols, ["Štatistika", "Popis"], [4*cm, 13*cm]))
    story.append(note(
        "Štatistiky sa počítajú len z úspešne spracovaných obrazov (stav OK). "
        "Metriky: dx [px], dy [px], dx [mm], dy [mm], Uhol [°], Spoľahlivosť."
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 11. VÝSTUPNÉ FORMÁTY
    # ══════════════════════════════════════════════════════════════════
    story += h1("11. Výstupné formáty")

    story += h2("CSV — hodnoty oddelené čiarkou")
    story.append(body("Každý riadok zodpovedá jednému obrazu. Stĺpce:"))
    csv_cols = [
        ["filename",    "Názov súboru obrazu"],
        ["dx_px",       "Horizontálny posun [px]"],
        ["dy_px",       "Vertikálny posun [px]"],
        ["dx_mm",       "Horizontálny posun [mm]"],
        ["dy_mm",       "Vertikálny posun [mm]"],
        ["angle_deg",   "Rotácia [°]"],
        ["confidence",  "Spoľahlivosť zarovnania [0, 1]"],
        ["status",      "OK alebo ERROR"],
    ]
    story.append(simple_table(csv_cols, ["Stĺpec", "Popis"], [4*cm, 13*cm]))
    story.append(spacer(0.4))

    story += h2("JSON — profil batch (so štatistikami)")
    story.append(code(
        '{\n'
        '  "results": [\n'
        '    {\n'
        '      "filename": "img001.png",\n'
        '      "status": "OK",\n'
        '      "dx_px": 2.34,\n'
        '      "dy_px": -1.12,\n'
        '      "dx_mm": 0.117,\n'
        '      "dy_mm": -0.056,\n'
        '      "angle_deg": 0.23,\n'
        '      "confidence": 0.91\n'
        '    }\n'
        '  ],\n'
        '  "stats": {\n'
        '    "count_total": 20,\n'
        '    "count_ok": 19,\n'
        '    "count_error": 1,\n'
        '    "dx_px_mean": 2.1, "dx_px_std": 0.3, "dx_px_min": 1.8, "dx_px_max": 2.6,\n'
        '    ...\n'
        '  }\n'
        '}'
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 12. TECHNICKÉ LIMITY
    # ══════════════════════════════════════════════════════════════════
    story += h1("12. Technické limity a tolerancie")

    story += h2("Pohybové limity algoritmov")
    story.append(body(
        "Oba algoritmy (ECC aj POC) sú navrhnuté pre <b>malý pohyb</b>. "
        "Pri prekročení limitov môže algoritmus konvergovať k nesprávnemu riešeniu."
    ))
    lim_rows = [
        ["Maximálny posun",    "~15 pixelov",     "Pri väčšom posune ECC nemusí nájsť správne riešenie"],
        ["Maximálna rotácia",  "~2°",             "Pre väčšie rotácie je potrebný hrubý odhad polohy"],
        ["Spoľahlivosť OK",    "> 0.70",          "Odporúčaný minimálny prah pre akceptovanie výsledku"],
        ["Spoľahlivosť slabá", "0.60 – 0.70",     "Výsledok je možný, ale neistý — overiť manuálne"],
        ["Spoľahlivosť zlá",   "&lt; 0.60",       "Zarovnanie pravdepodobne zlyhalo"],
    ]
    story.append(simple_table(lim_rows, ["Parameter", "Hodnota", "Popis"],
                               [4.5*cm, 3.5*cm, 9*cm]))
    story.append(spacer(0.4))

    story += h2("Testovacie tolerancie")
    story.append(body(
        "Automatizované testy overujú presnosť systému na syntetických dátach. "
        "Tolerancie sú definované v <b>tests/constants.py</b>:"
    ))
    tol_rows = [
        ["TRANSLATION_TOL_PX",  "0.10 px",  "Maximálna chyba dx/dy pre akceptovanie testu"],
        ["ROTATION_TOL_DEG",    "0.05°",    "Maximálna chyba uhla"],
        ["MM_CONVERSION_TOL",   "1e-5",     "Relatívna chyba konverzie px → mm"],
        ["MIN_CONFIDENCE",      "0.60",     "Minimálna spoľahlivosť pre OK výsledok v testoch"],
    ]
    story.append(simple_table(tol_rows, ["Konštanta", "Hodnota", "Popis"],
                               [5*cm, 3*cm, 9*cm]))
    story.append(spacer(0.4))

    story += h2("Podporované formáty obrazov")
    fmt_rows = [
        [".png",  "Odporúčaný — bezstratový, podporuje alfa kanál"],
        [".jpg / .jpeg", "Stratová kompresia — môže znížiť presnosť"],
        [".bmp",  "Bezstratový, väčšie súbory"],
        [".tiff / .tif", "Bezstratový, vhodný pre vedecké a priemyselné aplikácie"],
    ]
    story.append(simple_table(fmt_rows, ["Formát", "Poznámka"], [3.5*cm, 13.5*cm]))
    story.append(spacer(0.4))

    story += h2("Výstupné hodnoty zarovnania")
    out_rows = [
        ["dx_px",       "float", "Horizontálny posun v pixeloch (+ = doprava)"],
        ["dy_px",       "float", "Vertikálny posun v pixeloch (+ = nadol)"],
        ["angle_deg",   "float", "Rotácia v stupňoch (+ = hodinové ručičky)"],
        ["confidence",  "float", "ECC korelačná hodnota [0, 1] — miera spoľahlivosti"],
        ["ncc_score",   "float", "Normalizovaná krížová korelácia [-1, 1] po zarovnaní"],
        ["dx_mm",       "float", "Horizontálny posun v mm = dx_px × scale_mm_per_px"],
        ["dy_mm",       "float", "Vertikálny posun v mm = dy_px × scale_mm_per_px"],
    ]
    story.append(simple_table(out_rows, ["Pole", "Typ", "Popis"], [3.5*cm, 2*cm, 11.5*cm]))

    # ══════════════════════════════════════════════════════════════════
    # BUILD
    # ══════════════════════════════════════════════════════════════════
    doc.build(story, onFirstPage=title_page_bg, onLaterPages=title_page_bg)
    print(f"PDF vytvorene: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_document()
