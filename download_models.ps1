# Script to download models from Hugging Face
# This script downloads the required models for the bias detection system

Write-Host "Starting model download from Hugging Face" -ForegroundColor Green

# Create models directory if it doesn't exist
if (-not (Test-Path "models")) {
    Write-Host "Creating models directory" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "models" -Force | Out-Null
}

# Download DeBERTa model
Write-Host "`nDownloading DeBERTa model" -ForegroundColor Cyan
hf download chef2121/deberta-v2-political-stance-lora --local-dir models/deberta_model

# Download Longformer model
Write-Host "`nDownloading Longformer model" -ForegroundColor Cyan
hf download onioncult/article-longformer-finetuned --local-dir models/longformer-finetuned-model

Write-Host "`nModel download complete!" -ForegroundColor Green
Write-Host "Models saved to:" -ForegroundColor Yellow
Write-Host "  - models/deberta_model" -ForegroundColor White
Write-Host "  - models/longformer-finetuned-model" -ForegroundColor White
