#!/bin/bash
echo "Setting up Health-Care-Analysis project..."

# Make scripts executable
chmod +x setup_backend.sh setup_ai.sh setup_frontend.sh

# Setup backend
echo "1. Setting up Backend..."
./setup_backend.sh

# Setup AI engine
echo "2. Setting up AI Engine..."
./setup_ai.sh

# Setup frontend
echo "3. Setting up Frontend..."
./setup_frontend.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the full application:"
echo "1. Start Backend: cd health-backend-java/health-backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "2. Start Frontend: cd FrontEnd && npm run dev"
echo "3. Test AI Engine: cd AI && source venv/bin/activate && python sample_usage.py"