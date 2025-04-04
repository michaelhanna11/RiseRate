import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import requests
import os
import io
from datetime import datetime
import streamlit as st
import warnings

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Program metadata
PROGRAM_VERSION = "1.0 - 2025"
PROGRAM = "Rise Rate Calculator to AS 3610.2:2023"
COMPANY_NAME = "tekhne Consulting Engineers"
COMPANY_ADDRESS = "   "  # Update with actual address if needed
LOGO_URL = "https://drive.google.com/uc?export=download&id=1VebdT2loVGX57noP9t2GgQhwCNn8AA3h"
FALLBACK_LOGO_URL = "https://onedrive.live.com/download?cid=A48CC9068E3FACE0&resid=A48CC9068E3FACE0%21s252b6fb7fcd04f53968b2a09114d33ed"

def calculate_rate_of_rise(Pmax, D, H_form, T, C1, C2):
    K = (36 / (T + 16))**2
    def pressure_equation(R):
        if R <= 0:
            return 1e6
        term1 = C1 * np.sqrt(R)
        if H_form <= term1:
            return D * H_form - Pmax
        return D * (term1 + C2 * K * np.sqrt(H_form - term1)) - Pmax
    R_guess = max(0.1, (Pmax / D - C2 * K * np.sqrt(H_form)) / C1)**2 if (Pmax / D - C2 * K * np.sqrt(H_form)) > 0 else 0.1
    R_solution, info, ier, msg = fsolve(pressure_equation, R_guess, xtol=1e-8, maxfev=2000, full_output=True)
    return R_solution[0] if 0 < R_solution[0] <= 10 else float('nan')

def build_elements(inputs, max_R, y_max, project_number, project_name):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=14, spaceAfter=8, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle(name='SubtitleStyle', parent=styles['Normal'], fontSize=10, spaceAfter=8, alignment=TA_CENTER)
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=12, spaceAfter=6)
    normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=9, spaceAfter=6)
    table_header_style = ParagraphStyle(name='TableHeaderStyle', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=TA_LEFT)
    table_cell_style = ParagraphStyle(name='TableCellStyle', parent=styles['Normal'], fontSize=8, alignment=TA_LEFT, leading=8)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ])

    elements = []
    logo_file = "logo.png"
    if not os.path.exists(logo_file):
        for url in [LOGO_URL, FALLBACK_LOGO_URL]:
            try:
                response = requests.get(url, stream=True, allow_redirects=True, timeout=10)
                if 'image' in response.headers.get('Content-Type', '').lower():
                    response.raise_for_status()
                    with open(logo_file, 'wb') as f:
                        f.write(response.content)
                    break
            except Exception:
                continue

    company_text = f"<b>{COMPANY_NAME}</b><br/>{COMPANY_ADDRESS}"
    company_paragraph = Paragraph(company_text, normal_style)
    logo = Image(logo_file, width=50*mm, height=20*mm) if os.path.exists(logo_file) else Paragraph("[Logo Placeholder]", normal_style)
    header_data = [[logo, company_paragraph]]
    header_table = Table(header_data, colWidths=[60*mm, 120*mm])
    header_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('ALIGN', (1, 0), (1, 0), 'CENTER')]))
    elements.extend([header_table, Spacer(1, 4*mm), Paragraph("Rise Rate Calculation Report to AS 3610.2:2023", title_style)])

    project_details = f"Project Number: {project_number}<br/>Project Name: {project_name}<br/>Date: {datetime.now().strftime('%B %d, %Y')}"
    elements.extend([Paragraph(project_details, subtitle_style), Spacer(1, 2*mm), Paragraph("Input Parameters", heading_style)])

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
    input_data_formatted = [[Paragraph(row[0], table_header_style if i == 0 else table_cell_style),
                             Paragraph(row[1], table_header_style if i == 0 else table_cell_style)] for i, row in enumerate(input_data)]
    input_table = Table(input_data_formatted, colWidths=[100*mm, 80*mm])
    input_table.setStyle(table_style)
    elements.extend([input_table, Spacer(1, 4*mm), Paragraph("Rate of Rise vs Temperature Graph", heading_style)])
    graph_image = Image('graph.png', width=160*mm, height=120*mm) if os.path.exists('graph.png') else Paragraph("[Graph Placeholder]", normal_style)
    elements.append(graph_image)
    return elements

def generate_pdf_report(inputs, max_R, y_max, project_number, project_name):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    elements = build_elements(inputs, max_R, y_max, project_number, project_name)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 10)
        page_num = canvas.getPageNumber()
        canvas.drawCentredString(doc.pagesize[0] / 2.0, 10 * mm, f"{PROGRAM} {PROGRAM_VERSION} | tekhne © | Page {page_num}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def main():
    st.markdown(
        """
        <h1 style='font-size: 24px; margin-top: 0; margin-bottom: 10px;'>
            Rise Rate Calculator to AS 3610.2:2023
        </h1>
        """,
        unsafe_allow_html=True
    )
    with st.form("input_form"):
        project_number = st.text_input("Project Number", "PRJ-001")
        project_name = st.text_input("Project Name", "Sample Project")
        inputs = {
            'D': st.number_input("Wet Concrete Density (kN/m³)", min_value=0.0, max_value=30.0, value=25.0),
            'T_min': st.number_input("Min Temperature (°C)", min_value=5.0, max_value=30.0, value=5.0),
            'T_max': st.number_input("Max Temperature (°C)", min_value=5.0, max_value=30.0, value=30.0),
            'H_concrete': st.number_input("Total Concrete Height (m)", min_value=0.0, max_value=50.0, value=3.0),
            'H_form': st.number_input("Total Formwork Height (m)", min_value=0.0, max_value=50.0, value=3.3),
            'C2': st.selectbox("C2 Coefficient (per AS 3610.2:2023)", options=[0.3, 0.45, 0.6, 0.75], index=1),  # Default to 0.45
            'W': st.number_input("Plan Width (m)", min_value=0.0, max_value=100.0, value=2.0),
            'L': st.number_input("Plan Length (m)", min_value=0.0, max_value=100.0, value=3.0),
            'Pmax': st.number_input("Maximum Concrete Pressure (kN/m²)", min_value=0.0, value=60.0),
        }
        # Add checkbox for showing all C2 lines
        show_all_c2 = st.checkbox("Show comparison lines for all C2 coefficients", value=False)
        submitted = st.form_submit_button("Calculate")

    if submitted:
        inputs['C1'] = 1.5 if inputs['W'] <= 1.0 or inputs['L'] <= 1.0 else 1.0
        inputs['structure_type'] = "column" if inputs['C1'] == 1.5 else "wall"
        st.write(f"Assumed structure type: {inputs['structure_type']} (C1 = {inputs['C1']})")

        # Validation
        if not (5 <= inputs['T_min'] <= inputs['T_max'] <= 30):
            st.error("Temperatures must be between 5 and 30°C, with T_min <= T_max")
            return
        if not (0 < inputs['H_concrete'] <= inputs['H_form'] <= 50):
            st.error("Heights must be positive, with concrete height <= formwork height <= 50 m")
            return
        if not (0 < inputs['Pmax'] <= inputs['D'] * inputs['H_form']):
            st.error(f"Pmax must be between 0 and hydrostatic limit ({inputs['D'] * inputs['H_form']:.2f} kN/m²)")
            return

        T_range = np.linspace(inputs['T_min'], inputs['T_max'], 50)
        
        # Initialize plot
        plt.figure(figsize=(10, 6))
        
        # Define color palette and line styles
        colors = ['#00A859', '#FF5733', '#3498DB', '#9B59B6']  # Green, Red, Blue, Purple
        c2_values = [0.3, 0.45, 0.6, 0.75]
        line_styles = ['-', '--', ':', '-.']
        
        # Plot the selected C2 value (always shown)
        R_values = [calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], inputs['C2']) 
                   for T in T_range]
        plt.plot(T_range, R_values, color=colors[c2_values.index(inputs['C2'])], 
                linestyle='-', linewidth=2.5, 
                label=f'Selected (C2={inputs["C2"]}, Pmax={inputs["Pmax"]} kN/m²)')
        
        # Plot other C2 values if requested
        if show_all_c2:
            for c2 in [val for val in c2_values if val != inputs['C2']]:
                R_values_alt = [calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2) 
                              for T in T_range]
                plt.plot(T_range, R_values_alt, color=colors[c2_values.index(c2)], 
                        linestyle=line_styles[c2_values.index(c2)], linewidth=1.5, alpha=0.8,
                        label=f'C2={c2}')

        # Find maximum R value for y-axis scaling
        all_R = R_values.copy()
        if show_all_c2:
            for c2 in [val for val in c2_values if val != inputs['C2']]:
                all_R.extend([calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2) 
                            for T in T_range])
        max_R = np.nanmax(all_R)
        y_max = max_R * 1.2 if not np.isnan(max_R) else 10  # Fallback if all NaN
        
        # Plot formatting
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Rate of Rise (m/hr)', fontsize=12)
        plt.title(f'Rate of Rise vs Temperature - {project_name}', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, framealpha=1)
        plt.ylim(0, y_max)
        
        # Annotate points (only for selected C2)
        for T in np.arange(inputs['T_min'], inputs['T_max'] + 1, 5):
            if T <= inputs['T_max']:
                R = calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], inputs['C2'])
                if not np.isnan(R):
                    plt.text(T, R + max_R * 0.02, f'{R:.2f}', fontsize=9, ha='center', va='bottom', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        plt.tight_layout()
        plt.savefig('graph.png', dpi=600, bbox_inches='tight')
        st.image('graph.png')
        plt.close()

        pdf_data = generate_pdf_report(inputs, max_R, y_max, project_number, project_name)
        if pdf_data:
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"Rise_Rate_Calculation_Report_{project_name.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            st.success(f"Maximum calculated rate of rise: {max_R:.2f} m/hr")

if __name__ == "__main__":
    main()
