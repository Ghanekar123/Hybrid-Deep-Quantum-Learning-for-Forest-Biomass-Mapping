# MDQ-BioMap
## Multimodal Deep-Quantum Biomass Mapping

MDQ-BioMap is a multimodal machine-learning framework for above-ground biomass (AGB) estimation using optical, vegetation-index, SAR, LiDAR, terrain, and environmental features.

## Framework

The implementation contains:

1. Data preprocessing and train/validation/test splitting
2. Multimodal Transformer feature fusion
3. Quantum regression using QSVR and VQR
4. Classical machine-learning and deep-learning baselines
5. Cross-domain transfer evaluation
6. Limited-field-data learning curves
7. Validation plots
8. Wall-to-wall biomass raster mapping

## Project Structure

```text
MDQ-BioMap/
├── main.py
├── config.py
├── data_loader.py
├── transformer.py
├── quantum_models.py
├── classical_models.py
├── transfer_learning.py
├── learning_curves.py
├── visualization.py
├── raster_mapping.py
├── requirements.txt
├── README.md
