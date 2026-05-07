```bash
# Extract Dataset
unzip Data.zip -d ./Data/

# Download MobileSAM Weights
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# Install Dependencies
pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Run Pipeline
python -m src.driver --sam-ckpt mobile_sam.pt

# Fast Debug Mode
python -m src.driver --sam-ckpt mobile_sam.pt --fast
