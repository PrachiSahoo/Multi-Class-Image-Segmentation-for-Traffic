import numpy as np

CLASS_NAMES = [
    "Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", 
    "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", 
    "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", 
    "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", 
    "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", 
    "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLORS = np.array([
    [64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128],
    [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192],
    [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128],
    [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128],
    [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128],
    [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0]
], dtype=np.uint8)

ROAD_CLASS_IDS = [17]                          
PEDESTRIAN_CLASS_IDS = [2, 7, 16]               
VEHICLE_CLASS_IDS = [5, 13, 22, 25, 27]  
