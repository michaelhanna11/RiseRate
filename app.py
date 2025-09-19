import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import requests
import io
from datetime import datetime
import streamlit as st
import warnings
from PIL import Image as PilImage # Added Pillow import

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Program metadata
PROGRAM_INFO = {
    "version": "2.0 - 2025",
    "program_name": "Rise Rate Calculation to AS 3610.2:2023",
    "company_name": "tekhne Consulting Engineers",
    "company_address": "8/104-110 Mount St, North Sydney, NSW 2060"
}
LOGO_URL = "https://drive.google.com/uc?export=download&id=1VebdT2loVGX57noP9t2GgQhwCNn8AA3h"
FALLBACK_LOGO_URL = "https://onedrive.live.com/download?cid=A48CC9068E3FACE0&resid=A48CC9068E3FACE0%21s252b6fb7fcd04f53968b2a09114d33ed"

@st.cache_data
def calculate_rate_of_rise(Pmax, D, H_form, T, C1, C2):
    """
    Calculates the rate of rise (R) based on AS 3610.2:2023, Annex B.
    """
    K = (36 / (T + 16))**2
    
    def pressure_equation(R):
        if R <= 0 or (H_form - C1 * np.sqrt(R)) < 0:
            return 1e9
        return D * (C1 * np.sqrt(R) + C2 * K * np.sqrt(H_form - C1 * np.sqrt(R))) - Pmax
    
    R_guess = max(0.2, (Pmax / D - C2 * K * np.sqrt(H_form)) / C1)**2 if (Pmax / D - C2 * K * np.sqrt(H_form)) > 0 else 0.2
    R_solution, info, ier, msg = fsolve(pressure_equation, R_guess, full_output=True)
    
    if ier == 1:
        result = R_solution[0]
        final_R = np.clip(result, 0.2, 10.0)
        return final_R
    else:
        return float('nan')

def get_company_logo():
    """
    Downloads the company logo from a URL.
    """
    try:
        for url in [LOGO_URL, FALLBACK_LOGO_URL]:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                return io.BytesIO(response.content)
    except Exception:
        pass
    return None

def create_graph_image(inputs, project_name, show_all_c2):
    """
    Generates a graph of Rise Rate vs. Temperature.
    Includes the company logo on the graph itself.
    """
    T_range = np.linspace(inputs['T_min'], inputs['T_max'], 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#00A859', '#FF5733', '#3498DB', '#9B59B6']
    c2_values = [0.3, 0.45, 0.6, 0.75]
    
    # Plot selected C2 with a thicker solid line
    R_values = [calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], inputs['C2']) 
                for T in T_range]
    ax.plot(T_range, R_values, color=colors[c2_values.index(inputs['C2'])], 
            linestyle='-', linewidth=3.0, 
            label=f'Selected (C2={inputs["C2"]}, Pmax={inputs["Pmax"]} kN/m²)')
    
    if show_all_c2:
        for c2 in [val for val in c2_values if val != inputs['C2']]:
            R_values_alt = [calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2) 
                            for T in T_range]
            ax.plot(T_range, R_values_alt, color=colors[c2_values.index(c2)], 
                    linestyle='-', linewidth=2.0, alpha=0.8,
                    label=f'C2={c2}')

    all_R = R_values.copy()
    if show_all_c2:
        for c2 in [val for val in c2_values if val != inputs['C2']]:
            all_R.extend([calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2) 
                          for T in T_range])
    
    max_R = np.nanmax(all_R) if not np.all(np.isnan(all_R)) else 10
    y_max = max_R * 1.2 if not np.isnan(max_R) else 10
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Rate of Rise (m/hr)', fontsize=12)
    ax.set_title(f'Rate of Rise vs Temperature - {project_name}', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, framealpha=1)
    ax.set_ylim(0, y_max)
    
    # Add text labels and markers for all plotted lines at fixed temperature intervals
    plotted_c2_values = [inputs['C2']]
    if show_all_c2:
        plotted_c2_values.extend([val for val in c2_values if val != inputs['C2']])

    for c2 in plotted_c2_values:
        for T in np.arange(inputs['T_min'], inputs['T_max'] + 1, 5):
            if T <= inputs['T_max']:
                R = calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2)
                if not np.isnan(R):
                    # Add data point marker
                    marker_style = 'o' if c2 == inputs['C2'] else 's'
                    marker_color = colors[c2_values.index(c2)]
                    ax.plot(T, R, marker=marker_style, color=marker_color, markersize=8)
                    # Add text label
                    ax.text(T, R + max_R * 0.02, f'{R:.2f}', fontsize=9, ha='center', va='bottom', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # --- Corrected code for logo placement and size using PIL ---
    logo_data = get_company_logo()
    if logo_data:
        # Load and resize the image using PIL
        pil_img = PilImage.open(logo_data)
        new_width = 100 # Desired width in pixels
        aspect_ratio = pil_img.height / pil_img.width
        new_height = int(new_width * aspect_ratio)
        resized_img = pil_img.resize((new_width, new_height), PilImage.Resampling.LANCZOS)
        
        # Convert the resized image back to a format matplotlib can use
        resized_img_bytes = io.BytesIO()
        resized_img.save(resized_img_bytes, format='PNG')
        resized_img_bytes.seek(0)
        
        # Use figimage with the resized image data
        fig.figimage(plt.imread(resized_img_bytes),
                     xo=fig.bbox.width - new_width - 20,  # 20 pixels padding from right edge
                     yo=20,  # 20 pixels padding from bottom edge
                     origin='upper',
                     zorder=10,
                     alpha=0.7)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf, max_R, y_max

def build_pdf_elements(inputs, max_R, y_max, project_number, project_name, graph_buffer, logo_buffer):
    """
    Builds the list of elements for the PDF report.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=14, spaceAfter=8, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle(name='SubtitleStyle', parent=styles['Normal'], fontSize=10, spaceAfter=8, alignment=TA_CENTER)
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=12, spaceAfter=6, fontName='Helvetica-Bold')
    normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=9, spaceAfter=6)
    table_header_style = ParagraphStyle(name='TableHeaderStyle', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=TA_CENTER)
    table_cell_style = ParagraphStyle(name='TableCellStyle', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, leading=8)
    
    table_style_main = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F8FF')),
    ])

    elements = []
    
    # Updated Header Table
    company_info = [
        [Paragraph("<b>tekhne Consulting Engineers</b>", styles['Normal']), RLImage(logo_buffer, width=50*mm, height=20*mm) if logo_buffer else Paragraph("[Logo Placeholder]", styles['Normal'])],
        [Paragraph(PROGRAM_INFO['company_address'], styles['Normal']), ""]
    ]
    header_table = Table(company_info, colWidths=[120*mm, 60*mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (0, -1), 0),
        ('RIGHTPADDING', (1, 0), (1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LINEBELOW', (0, -1), (0, -1), 0.5, colors.black),
    ]))

    elements.extend([
        header_table,
        Spacer(1, 2*mm),
        Paragraph(f"Report: {PROGRAM_INFO['program_name']}", title_style),
        Paragraph(f"Project Number: {project_number} | Project Name: {project_name}<br/>Date: {datetime.now().strftime('%B %d, %Y')}", subtitle_style),
        Spacer(1, 2*mm),
        Paragraph("Input Parameters", heading_style)
    ])

    input_data = [
        ["Parameter", "Value"],
        ["Wet Concrete Density (kN/m³)", f"{inputs['D']:.2f}"],
        ["Min Temperature (°C)", f"{inputs['T_min']:.1f}"],
        ["Max Temperature (°C)", f"{inputs['T_max']:.1f}"],
        ["Total Concrete Height (m)", f"{inputs['H_concrete']:.2f}"],
        ["Total Formwork Height (m)", f"{inputs['H_form']:.2f}"],
        ["C2 Coefficient", f"{inputs['C2']:.2f}"],
        ["Plan Width (m)", f"{inputs['W']:.2f}"],
        ["Plan Length (m)", f"{inputs['L']:.2f}"],
        ["Maximum Concrete Pressure (kN/m²)", f"{inputs['Pmax']:.2f}"],
        ["Structure Type (C1)", f"{inputs['structure_type']} ({inputs['C1']:.1f})"],
    ]
    
    input_table = Table(input_data, colWidths=[100*mm, 80*mm])
    input_table.setStyle(table_style_main)
    
    elements.extend([
        input_table,
        Spacer(1, 2*mm),
        Paragraph("Rate of Rise vs Temperature Graph", heading_style)
    ])
    
    if graph_buffer:
        graph_image = RLImage(graph_buffer, width=150*mm, height=90*mm)
    else:
        graph_image = Paragraph("[Graph Placeholder]", normal_style)
    
    elements.append(graph_image)
    
    return elements

def generate_pdf_report(inputs, max_R, y_max, project_number, project_name, graph_buffer, logo_buffer):
    """
    Generates the final PDF document.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, 
                            leftMargin=15*mm, rightMargin=15*mm, 
                            topMargin=10*mm, bottomMargin=10*mm)
    
    elements = build_pdf_elements(inputs, max_R, y_max, project_number, project_name, graph_buffer, logo_buffer)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 10)
        page_num = canvas.getPageNumber()
        canvas.drawCentredString(doc.pagesize[0] / 2.0, 10 * mm, 
                                 f"{PROGRAM_INFO['program_name']} {PROGRAM_INFO['version']} | tekhne © | Page {page_num}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    pdf_buffer.seek(0)
    return pdf_buffer

def main():
    """
    Main Streamlit application function.
    """
    st.set_page_config(page_title="Rise Rate Calculator - AS 3610.2:2023 | CIRIA Report 108")
    
    st.markdown(
        f"""
        <h1 style='font-size: 24px; margin-top: 0; margin-bottom: 10px;'>
            {PROGRAM_INFO['program_name']}
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    with st.form("input_form"):
        project_number = st.text_input("Project Number", "TK-2025001")
        project_name = st.text_input("Project Name", "Project")
        
        inputs = {
            'D': st.number_input("Wet Concrete Density (kN/m³)", min_value=0.0, max_value=30.0, value=25.0, help="Density of the fresh concrete."),
            'T_min': st.number_input("Min Temperature (°C)", min_value=5.0, max_value=30.0, value=5.0, help="Minimum fresh concrete temperature."),
            'T_max': st.number_input("Max Temperature (°C)", min_value=5.0, max_value=30.0, value=30.0, help="Maximum fresh concrete temperature."),
            'H_concrete': st.number_input("Total Concrete Height (m)", min_value=0.0, max_value=50.0, value=3.0, help="The total height of the concrete pour."),
            'H_form': st.number_input("Total Formwork Height (m)", min_value=0.0, max_value=50.0, value=3.3, help="The total height of the formwork system."),
            'C2': st.selectbox(
                "C2 Coefficient (per AS 3610.2:2023)",
                options=[0.3, 0.45, 0.6, 0.75],
                index=1,
                help="C2 accounts for concrete properties. Refer to AS 3610.2 Table 2.1 for values. 0.45 is a common default for normal concrete."
            ),
            'W': st.number_input("Plan Width (m)", min_value=0.0, max_value=100.0, value=2.0, help="The width of the formwork in plan."),
            'L': st.number_input("Plan Length (m)", min_value=0.0, max_value=100.0, value=3.0, help="The length of the formwork in plan."),
            'Pmax': st.number_input("Maximum Concrete Pressure (kN/m²)", min_value=0.0, value=60.0, help="The maximum design pressure the formwork can withstand.")
        }
        
        show_all_c2 = st.checkbox("Show comparison lines for all C2 coefficients", value=False)
        submitted = st.form_submit_button("Calculate")

    if submitted:
        inputs['C1'] = 1.5 if inputs['W'] < 2.0 and inputs['L'] < 2.0 else 1.0
        inputs['structure_type'] = "column" if inputs['C1'] == 1.5 else "wall"
        st.info(f"The program has assumed a structure type of **{inputs['structure_type']}** (C1 = **{inputs['C1']}**).")

        # Validation
        if not (5 <= inputs['T_min'] <= inputs['T_max'] <= 30):
            st.error("Temperatures must be between 5 and 30°C, with T_min <= T_max.")
            return
        if not (0 < inputs['H_concrete'] <= inputs['H_form'] <= 50):
            st.error("Heights must be positive, with concrete height <= formwork height <= 50 m.")
            return
        if not (0 < inputs['Pmax'] <= inputs['D'] * inputs['H_form']):
            st.error(f"Pmax must be between 0 and the hydrostatic limit ({inputs['D'] * inputs['H_form']:.2f} kN/m²).")
            return

        try:
            # Create graph and get max values
            graph_buffer, max_R, y_max = create_graph_image(inputs, project_name, show_all_c2)
            
            # Display graph in Streamlit
            st.image(graph_buffer)
            
            # Get company logo
            logo_buffer = get_company_logo()
            
            # Generate PDF
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_pdf_report(inputs, max_R, y_max, project_number, project_name, graph_buffer, logo_buffer)
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"Rise_Rate_Calculation_Report_{project_name.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

                st.download_button(
                    label="Download Graph as PNG",
                    data=graph_buffer,
                    file_name=f"Rise_Rate_Graph_{project_name.replace(' ', '_')}.png",
                    mime="image/png"
                )
                
                st.success(f"Maximum calculated rate of rise for your selected parameters is: **{max_R:.2f} m/hr**")
                st.info("Developed by Michael Hanna on behald of Tekhne Consulting Engineers")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
