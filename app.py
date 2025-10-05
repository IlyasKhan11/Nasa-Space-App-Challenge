from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import numpy as np
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline
import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Create FastAPI app
app = FastAPI(title="NASA Exoplanet Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default port
        "http://127.0.0.1:3000",  # React alternative
        "http://localhost:3001",  # Alternative React port
        "http://127.0.0.1:3001",  # Alternative React port
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173",  # Vite alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# ✅ Fixed: Pydantic v2 compatible model
class ExoplanetData(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_model_snr: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float
    koi_kepmag: float
    koi_fpflag_nt: int
    koi_fpflag_ss: int
    koi_fpflag_co: int
    koi_fpflag_ec: int

    # ✅ Validate that numeric values are positive
    @field_validator(
        "koi_period", "koi_duration", "koi_depth", "koi_prad",
        "koi_teq", "koi_insol", "koi_model_snr", "koi_steff",
        "koi_slogg", "koi_srad", "koi_kepmag"
    )
    def check_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    # ✅ Validate that flags are binary (0 or 1)
    @field_validator("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec")
    def check_binary_flags(cls, v):
        if v not in [0, 1]:
            raise ValueError("Flag must be 0 or 1")
        return v

# Utility: convert to DataFrame
class CustomData:
    def __init__(self, data: ExoplanetData):
        self.data = data
    
    def get_data_as_data_frame(self):
        return pd.DataFrame([self.data.dict()])

# -------------------------
# Web routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predictdata", response_class=HTMLResponse)
async def predict_datapoint_get(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint_post(
    request: Request,
    koi_period: float = Form(...),
    koi_duration: float = Form(...),
    koi_depth: float = Form(...),
    koi_prad: float = Form(...),
    koi_teq: float = Form(...),
    koi_insol: float = Form(...),
    koi_model_snr: float = Form(...),
    koi_steff: float = Form(...),
    koi_slogg: float = Form(...),
    koi_srad: float = Form(...),
    koi_kepmag: float = Form(...),
    koi_fpflag_nt: int = Form(...),
    koi_fpflag_ss: int = Form(...),
    koi_fpflag_co: int = Form(...),
    koi_fpflag_ec: int = Form(...)
):
    try:
        exoplanet_data = ExoplanetData(
            koi_period=koi_period,
            koi_duration=koi_duration,
            koi_depth=koi_depth,
            koi_prad=koi_prad,
            koi_teq=koi_teq,
            koi_insol=koi_insol,
            koi_model_snr=koi_model_snr,
            koi_steff=koi_steff,
            koi_slogg=koi_slogg,
            koi_srad=koi_srad,
            koi_kepmag=koi_kepmag,
            koi_fpflag_nt=koi_fpflag_nt,
            koi_fpflag_ss=koi_fpflag_ss,
            koi_fpflag_co=koi_fpflag_co,
            koi_fpflag_ec=koi_fpflag_ec
        )

        custom_data = CustomData(exoplanet_data)
        pred_df = custom_data.get_data_as_data_frame()
        print("Input Data:", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        prediction_label = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}.get(results[0], "UNKNOWN")

        return templates.TemplateResponse("home.html", {
            "request": request, 
            "results": prediction_label,
            "input_data": exoplanet_data.dict()
        })

    except Exception as e:
        print("Prediction error:", e)
        return templates.TemplateResponse("home.html", {
            "request": request, 
            "error": f"Prediction error: {str(e)}"
        })

# -------------------------
# API endpoints
# -------------------------
@app.post("/api/predict")
async def api_predict(data: ExoplanetData):
    try:
        custom_data = CustomData(data)
        pred_df = custom_data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        prediction_label = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}.get(results[0], "UNKNOWN")

        return {
            "prediction": prediction_label,
            "prediction_code": int(results[0]),
            "input_data": data.dict()
        }

    except Exception as e:
        print("API prediction error:", e)
        return {"error": str(e)}

# -------------------------
# CSV Upload and Batch Prediction endpoints
# -------------------------
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            return {"error": "Only CSV files are allowed"}
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📊 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Validate required columns
        required_columns = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
            'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
            'koi_kepmag', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing required columns: {missing_columns}"}
        
        # Clean the data - handle missing values and infinite values
        df_clean = df[required_columns].copy()
        
        # Fill NaN values with appropriate defaults
        numeric_columns = [col for col in required_columns if col not in ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']]
        flag_columns = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        
        # Fill numeric NaNs with median values
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill flag NaNs with 0
        for col in flag_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0).astype(int)
        
        # Replace infinite values with large finite numbers
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        print(f"✅ Data cleaned. Making predictions on {len(df_clean)} rows...")
        
        # Make predictions
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(df_clean)
        
        # Add predictions to original dataframe (not the cleaned one for display)
        df['prediction'] = predictions
        df['prediction_label'] = df['prediction'].map({0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"})
        
        # Generate results CSV for download
        results_csv = df.to_csv(index=False)
        csv_filename = f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = f"temp/{csv_filename}"
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save CSV file
        with open(csv_path, 'w') as f:
            f.write(results_csv)
        
        # Generate PDF report (optional)
        pdf_filename = f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = f"temp/{pdf_filename}"
        generate_pdf_report(df, pdf_path)
        
        # Prepare response data - ensure all values are JSON serializable
        results_preview = df.head(10).replace([np.nan, np.inf, -np.inf], None).to_dict('records')
        
        # Convert numpy types to native Python types for JSON serialization
        for record in results_preview:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    record[key] = float(value) if not np.isnan(value) and not np.isinf(value) else None
                elif isinstance(value, np.bool_):
                    record[key] = bool(value)
        
        response_data = {
            "message": "Predictions completed successfully",
            "total_predictions": len(predictions),
            "prediction_summary": {
                "FALSE POSITIVE": int((predictions == 0).sum()),
                "CANDIDATE": int((predictions == 1).sum()),
                "CONFIRMED": int((predictions == 2).sum())
            },
            "csv_download_url": f"/download-csv/{csv_filename}",
            "pdf_download_url": f"/download-pdf/{pdf_filename}",
            "results_preview": results_preview
        }
        
        return response_data
        
    except Exception as e:
        print(f"CSV processing error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error processing CSV: {str(e)}"}
# CSV Download endpoint
@app.get("/download-csv/{filename}")
async def download_csv(filename: str):
    csv_path = f"temp/{filename}"
    if os.path.exists(csv_path):
        return FileResponse(
            csv_path, 
            media_type='text/csv', 
            filename=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    else:
        return {"error": "CSV file not found"}

# PDF Download endpoint
@app.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    pdf_path = f"temp/{filename}"
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path, 
            media_type='application/pdf', 
            filename=f"exoplanet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    else:
        return {"error": "PDF file not found"}

# Function to generate PDF report
def generate_pdf_report(df, pdf_path):
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("NASA Exoplanet Prediction Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary
        story.append(Paragraph("Prediction Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Summary table
        summary_data = [
            ['Total Predictions', str(len(df))],
            ['FALSE POSITIVE', str(int((df['prediction'] == 0).sum()))],
            ['CANDIDATE', str(int((df['prediction'] == 1).sum()))],
            ['CONFIRMED', str(int((df['prediction'] == 2).sum()))]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Detailed results (first 15 rows)
        story.append(Paragraph("Detailed Results (First 15 Predictions)", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Prepare data for table
        display_df = df.head(15)[['koi_period', 'koi_prad', 'koi_model_snr', 'prediction_label']].copy()
        display_df.columns = ['Period (days)', 'Planet Radius', 'SNR', 'Prediction']
        
        # Convert to table data
        table_data = [display_df.columns.tolist()] + display_df.values.tolist()
        
        # Create table
        results_table = Table(table_data, colWidths=[1.2*inch, 1.2*inch, 1*inch, 1.2*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(results_table)
        
        # Footer
        story.append(Spacer(1, 20))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1,
            textColor=colors.grey
        )
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
        story.append(Paragraph("NASA Exoplanet Prediction System", footer_style))
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {pdf_path}")
        
    except Exception as e:
        print(f"❌ PDF generation error: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "NASA Exoplanet Prediction API is running"}

# CORS preflight handler
@app.options("/api/{rest_of_path:path}")
async def preflight_handler():
    return {"message": "CORS preflight"}

@app.options("/api/upload-csv")
async def options_upload_csv():
    return {"message": "CORS preflight"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)