import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import requests
import os
import streamlit as st
import io

# Program version
PROGRAM_VERSION = "1.0 - 2025"
PROGRAM = "Rise Rate Calculator to AS 3610.2:2023"

# Company details
COMPANY_NAME = "Tekhne Consulting Engineers"
COMPANY_ADDRESS = "   "

# Google Drive logo URL
LOGO_URL = "https://drive.google.com/uc?export=download&id=1VebdT2loVGX57noP9t2GgQhwCNn8AA3h"
FALLBACK_LOGO_URL = "https://onedrive.live.com/download?cid=A48CC9068E3FACE0&resid=A48CC9068E3FACE0%21s252b6fb7fcd04f53968b2a09114d33ed"

def calculate_rate_of_rise(Pmax, D, H_form, T, C1, C2):
    K = (36 / (T + 16))**2
    def pressure_equation(R):
        if R <= 0:  # Handle negative or zero R
            return 1e6  # Large penalty
        term1 = C1 * np.sqrt(R)
        if H_form <= term1:
            return D * H_form - Pmax
        return D * (term1 + C2 * K * np.sqrt(H_form - term1)) - Pmax
    R_guess = max(0.1, ((Pmax / D - C2 * K * np.sqrt(H_form)) / C1)**2) if (Pmax / D - C2 * K * np.sqrt(H_form)) > 0 else 0.1
    R_solution, info, ier, msg = fsolve(pressure_equation, R_guess, xtol=1e-8, maxfev=2000, full_output=True)
    return R_solution[0] if 0 < R_solution[0] <= 10 else float('nan')

# Streamlit app
st.title("Rise Rate Calculator to AS 3610.2:2023")

# User inputs with Streamlit widgets
D = st.number_input("Wet concrete density (kN/m³)", min_value=0.1, max_value=30.0, value=25.0, step=0.1)
T_min = st.number_input("Min temperature (°C)", min_value=5.0, max_value=30.0, value=5.0, step=0.1)
T_max = st.number_input("Max temperature (°C)", min_value=5.0, max_value=30.0, value=30.0, step=0.1)
H_concrete = st.number_input("Total concrete height (m)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
H_form = st.number_input("Total formwork height (m)", min_value=0.1, max_value=50.0, value=3.3, step=0.1)
C2 = st.number_input("C2 coefficient", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
W = st.number_input("Plan width (m)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
L = st.number_input("Plan length (m)", min_value=0.1, max_value=100.0, value=3.0, step=0.1)
Pmax = st.number_input("Maximum concrete pressure (kN/m²)", min_value=0.1, max_value=1000.0, value=60.0, step=0.1)
Project = st.text_input("Project Name", value="__")

# Determine structure type
C1 = 1.5 if W <= 1.0 or L <= 1.0 else 1.0
structure_type = "column" if C1 == 1.5 else "wall"
st.write(f"Assumed structure type: {structure_type} (C1 = {C1})")

# Validate inputs
if not (0 < D <= 30):
    st.error("Density must be between 0 and 30 kN/m³")
elif not (5 <= T_min <= T_max <= 30):
    st.error("Temperatures must be between 5 and 30°C, with T_min <= T_max")
elif not (0 < H_concrete <= H_form <= 50):  # Fixed typo: >= to <=
    st.error("Heights must be positive, with concrete height <= formwork height <= 50 m")
elif not (0 < C2 <= 1.0):
    st.error("C2 coefficient must be between 0 and 1.0")
elif not (0 < W <= 100 and 0 < L <= 100):
    st.error("Plan dimensions must be between 0 and 100 m")
elif not (0 < Pmax <= D * H_form):
    st.error(f"Pmax must be between 0 and hydrostatic limit ({D * H_form:.2f} kN/m²)")
else:
    # Calculate R across temperature range
    T_range = np.linspace(T_min, T_max, 50)
    R_values = [calculate_rate_of_rise(Pmax, D, H_form, T, C1, C2) for T in T_range]

    # Debug: Show R at 5°C intervals
    st.subheader("Rate of Rise (R) at 5°C Intervals")
    for T in np.arange(T_min, T_max + 1, 5):
        if T <= T_max:
            R = calculate_rate_of_rise(Pmax, D, H_form, T, C1, C2)
            st.write(f"T = {T}°C, R = {R:.2f} m/hr" if not np.isnan(R) else f"T = {T}°C, R = NaN")

    # Determine maximum R
    max_R = np.nanmax(R_values)
    y_max = max_R * 1.1

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T_range, R_values, 'b-', label=f'Rate of Rise (Pmax = {Pmax} kN/m²)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Rate of Rise (m/hr)')
    ax.set_title(f'Rate of Rise vs Temperature - {Project}\nD = {D} kN/m³, C2 = {C2}, P= {Pmax} kN/m²')
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, y_max)
    T_steps = np.arange(T_min, T_max + 1, 5)
    for T in T_steps:
        if T <= T_max:
            R = calculate_rate_of_rise(Pmax, D, H_form, T, C1, C2)
            if not np.isnan(R):
                ax.text(T, R + max_R * 0.02, f'{R:.2f}', fontsize=10, ha='center', va='bottom', color='black')
    st.pyplot(fig)
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Download logo
    logo_file = "logo.png"
    for url in [LOGO_URL, FALLBACK_LOGO_URL]:
        try:
            response = requests.get(url, stream=True, allow_redirects=True)
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type.lower():
                raise ValueError(f"Received {content_type} instead of an image")
            response.raise_for_status()
            with open(logo_file, 'wb') as f:
                f.write(response.content)
            break
        except Exception as e:
            st.warning(f"Failed to download logo from {url}: {e}")
            logo_file = None
    else:
        st.warning("All URL attempts failed. Using placeholder.")
        logo_file = None

    # Generate PDF in memory
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    # Header: Logo, Company Name, Address
    if logo_file and os.path.exists(logo_file):
        c.drawImage(logo_file, 10 * mm, height - 20 * mm, width=50 * mm, height=20 * mm)
    else:
        c.setFont("Helvetica", 12)
        c.drawString(20 * mm, height - 30 * mm, "[Logo Placeholder]")

    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, height - 30 * mm, COMPANY_NAME)
    c.setFont("Helvetica", 10)
    c.drawCentredString(width / 2, height - 40 * mm, COMPANY_ADDRESS)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, height - 50 * mm, "Rate of rise calculation to AS 3610.2:2023")

    # Input values
    c.setFont("Helvetica", 10)
    text_y = height - 80 * mm
    c.drawString(20 * mm, text_y, "Input Parameters:")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Wet Concrete Density: {D} kN/m³")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Min Temperature: {T_min}°C")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Max Temperature: {T_max}°C")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Total Concrete Height: {H_concrete} m")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Total Formwork Height: {H_form} m")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    C2 Coefficient: {C2}")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Plan Width: {W} m")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Plan Length: {L} m")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Maximum Concrete Pressure: {Pmax} kN/m²")
    text_y -= 10 * mm
    c.drawString(30 * mm, text_y, f"    Structure Type: {structure_type} (C1 = {C1})")

    # Add graph
    c.drawImage('graph.png', 20 * mm, 20 * mm, width=160 * mm, height=100 * mm)

    # Footer
    c.setFont("Helvetica", 10)
    footer_text = f"{PROGRAM} {PROGRAM_VERSION} | tekhne ©"
    c.drawCentredString(width / 2, 10 * mm, footer_text)

    # Save PDF to buffer
    c.save()
    pdf_buffer.seek(0)

    # Download button for PDF
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"Rise Rate Calculation Report - {Project.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

    st.write(f"Maximum calculated rate of rise: {max_R:.2f} m/hr")
    st.write(f"Y-axis maximum set to: {y_max:.2f} m/hr")
