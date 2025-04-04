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
        # Add new checkbox for multiple C2 lines
        show_all_c2 = st.checkbox("Show all C2 coefficient lines on graph", value=False)
        submitted = st.form_submit_button("Calculate")

    if submitted:
        inputs['C1'] = 1.5 if inputs['W'] <= 1.0 or inputs['L'] <= 1.0 else 1.0
        inputs['structure_type'] = "column" if inputs['C1'] == 1.5 else "wall"
        st.write(f"Assumed structure type: {inputs['structure_type']} (C1 = {inputs['C1']})")

        # Validation (unchanged)
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
        
        # Define color palette for different C2 lines
        colors = ['#00A859', '#FF5733', '#3498DB', '#9B59B6']
        c2_values = [0.3, 0.45, 0.6, 0.75]
        
        # Plot the selected C2 value (always shown)
        R_values = [calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], inputs['C2']) 
                   for T in T_range]
        plt.plot(T_range, R_values, color=colors[c2_values.index(inputs['C2'])], 
                linestyle='-', linewidth=2, 
                label=f'Selected C2={inputs["C2"]} (Pmax={inputs["Pmax"]} kN/m²)')
        
        # Plot other C2 values if requested
        if show_all_c2:
            for c2 in [val for val in c2_values if val != inputs['C2']]:
                R_values_alt = [calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2) 
                              for T in T_range]
                plt.plot(T_range, R_values_alt, color=colors[c2_values.index(c2)], 
                        linestyle='--', alpha=0.7, 
                        label=f'C2={c2}')
        
        # Find maximum R value for y-axis scaling
        all_R = R_values.copy()
        if show_all_c2:
            for c2 in [val for val in c2_values if val != inputs['C2']]:
                all_R.extend([calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], c2) 
                            for T in T_range])
        max_R = np.nanmax(all_R)
        y_max = max_R * 1.1 if not np.isnan(max_R) else 10  # Fallback if all NaN
        
        # Plot formatting
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Rate of Rise (m/hr)')
        plt.title(f'Rate of Rise vs Temperature - {project_name}')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, y_max)
        
        # Annotate points (only for selected C2)
        for T in np.arange(inputs['T_min'], inputs['T_max'] + 1, 5):
            if T <= inputs['T_max']:
                R = calculate_rate_of_rise(inputs['Pmax'], inputs['D'], inputs['H_form'], T, inputs['C1'], inputs['C2'])
                if not np.isnan(R):
                    plt.text(T, R + max_R * 0.02, f'{R:.2f}', fontsize=10, ha='center', va='bottom', color='black')
        
        plt.savefig('graph.png', dpi=600, bbox_inches='tight')
        st.image('graph.png')
        plt.close()

        # Rest of the code remains unchanged...
        pdf_data = generate_pdf_report(inputs, max_R, y_max, project_number, project_name)
        if pdf_data:
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"Rise_Rate_Calculation_Report_{project_name.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            st.write(f"Maximum calculated rate of rise: {max_R:.2f} m/hr")
