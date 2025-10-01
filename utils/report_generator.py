import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import os
from datetime import datetime

class ReportGenerator:
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,
            textColor=colors.darkred
        )
        
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
    
    def create_visualizations(self, user_data, prediction_result):
        
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Health Parameters Analysis', fontsize=16, fontweight='bold')
            
            normal_ranges = {
                'Blood Pressure': (90, 120),
                'Cholesterol': (100, 200),
                'Heart Rate': (60, 100),
                'Age': (18, 65)
            }
            
            params_to_plot = ['trestbps', 'chol', 'thalach', 'age']
            param_names = ['Blood Pressure', 'Cholesterol', 'Heart Rate', 'Age']
            
            for i, (param, name) in enumerate(zip(params_to_plot, param_names)):
                ax = axes[i//2, i%2]
                
                if param in user_data:
                    user_value = user_data[param]
                    normal_min, normal_max = normal_ranges[name]
                    
                    categories = ['Normal Range', 'Your Value']
                    values = [normal_max, user_value]
                    colors_list = ['lightblue', 'red' if user_value > normal_max or user_value < normal_min else 'green']
                    
                    bars = ax.bar(categories, values, color=colors_list, alpha=0.7)
                    ax.set_title(f'{name} Comparison')
                    ax.set_ylabel('Value')
                    
                    ax.axhspan(normal_min, normal_max, alpha=0.3, color='green', label='Normal Range')
                    
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value}', ha='center', va='bottom')
                    
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'{name}\nData Not Available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{name} Comparison')
            
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            img_buffer = io.BytesIO()
            plt.close('all')
            return img_buffer
    
    def create_prediction_chart(self, prediction_probability):
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            labels = ['No Heart Disease', 'Heart Disease Risk']
            sizes = [1 - prediction_probability, prediction_probability]
            colors = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Heart Disease Risk Assessment', fontsize=14, fontweight='bold')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error creating prediction chart: {str(e)}")
            img_buffer = io.BytesIO()
            plt.close('all')
            return img_buffer
    
    def generate_pdf_report(self, user_data, prediction_result, prediction_probability, 
                          patient_info=None, extracted_text="", output_path="heart_disease_report.pdf"):
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            title = Paragraph("Heart Disease Risk Assessment Report", self.title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            if patient_info:
                subtitle = Paragraph("Patient Information", self.subtitle_style)
                story.append(subtitle)
                
                patient_name = patient_info.get('patient_name', 'N/A')
                patient_id = patient_info.get('patient_id', 'N/A')
                
                name_para = Paragraph(f"<b>Patient Name:</b> {patient_name}", self.normal_style)
                story.append(name_para)
                
                id_para = Paragraph(f"<b>Patient ID:</b> {patient_id}", self.normal_style)
                story.append(id_para)
                
                story.append(Spacer(1, 20))
            
            date_str = datetime.now().strftime("%B %d, %Y")
            date_para = Paragraph(f"Report Generated: {date_str}", self.normal_style)
            story.append(date_para)
            story.append(Spacer(1, 20))
            
            subtitle = Paragraph("Risk Assessment Result", self.subtitle_style)
            story.append(subtitle)
            
            risk_level = "HIGH RISK" if prediction_result == 1 else "LOW RISK"
            risk_color = colors.red if prediction_result == 1 else colors.green
            risk_style = ParagraphStyle(
                'RiskStyle',
                parent=self.normal_style,
                fontSize=14,
                textColor=risk_color,
                fontName='Helvetica-Bold'
            )
            
            risk_para = Paragraph(f"Risk Level: {risk_level}", risk_style)
            story.append(risk_para)
            
            prob_para = Paragraph(f"Risk Probability: {prediction_probability:.2%}", self.normal_style)
            story.append(prob_para)
            story.append(Spacer(1, 20))
            
            subtitle = Paragraph("Health Parameters", self.subtitle_style)
            story.append(subtitle)
            
            param_names = {
                'age': 'Age',
                'sex': 'Sex (0=Female, 1=Male)',
                'cp': 'Chest Pain Type',
                'trestbps': 'Resting Blood Pressure',
                'chol': 'Cholesterol',
                'fbs': 'Fasting Blood Sugar',
                'restecg': 'Resting ECG',
                'thalach': 'Maximum Heart Rate',
                'exang': 'Exercise Induced Angina',
                'oldpeak': 'ST Depression',
                'slope': 'Slope',
                'ca': 'Number of Major Vessels',
                'thal': 'Thalassemia'
            }
            
            for param, value in user_data.items():
                if param in param_names:
                    param_text = f"{param_names[param]}: {value}"
                    para = Paragraph(param_text, self.normal_style)
                    story.append(para)
            
            story.append(Spacer(1, 20))
            
            subtitle = Paragraph("Health Analysis Charts", self.subtitle_style)
            story.append(subtitle)
            
            comparison_img = self.create_visualizations(user_data, prediction_result)
            img = RLImage(comparison_img, width=6*inch, height=5*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            
            prediction_img = self.create_prediction_chart(prediction_probability)
            img2 = RLImage(prediction_img, width=4*inch, height=3*inch)
            story.append(img2)
            story.append(Spacer(1, 20))
            
            subtitle = Paragraph("Recommendations", self.subtitle_style)
            story.append(subtitle)
            
            recommendations = [
                "• Consult with a healthcare professional for detailed analysis",
                "• Maintain a healthy diet and regular exercise routine",
                "• Monitor blood pressure and cholesterol levels regularly",
                "• Avoid smoking and excessive alcohol consumption",
                "• Get regular health check-ups"
            ]
            
            if prediction_result == 1:
                recommendations.insert(0, "• HIGH RISK: Immediate consultation with a cardiologist recommended")
            
            for rec in recommendations:
                para = Paragraph(rec, self.normal_style)
                story.append(para)
            
            story.append(Spacer(1, 20))
            
            disclaimer = Paragraph(
                "Disclaimer: This report is for informational purposes only and should not replace professional medical advice.",
                ParagraphStyle('Disclaimer', parent=self.normal_style, fontSize=8, textColor=colors.grey)
            )
            story.append(disclaimer)
            
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

def create_sample_report():
    generator = ReportGenerator()
    
    sample_data = {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 140,
        'chol': 250,
        'fbs': 1,
        'restecg': 1,
        'thalach': 150,
        'exang': 1,
        'oldpeak': 2.5,
        'slope': 1,
        'ca': 2,
        'thal': 2
    }
    
    output_path = generator.generate_pdf_report(
        sample_data, 
        prediction_result=1, 
        prediction_probability=0.75,
        output_path="sample_report.pdf"
    )
    
    print(f"Sample report generated: {output_path}")

if __name__ == "__main__":
    create_sample_report()
