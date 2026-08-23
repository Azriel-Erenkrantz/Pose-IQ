import numpy as np

class AngleCalculator:
    KP = {
        'nose': 0, 
        'l_shoulder': 11, 'r_shoulder': 12, 
        'l_elbow': 13, 'r_elbow': 14,
        'l_wrist': 15, 'r_wrist': 16, 
        'l_hip': 23, 'r_hip': 24,
        'l_knee': 25, 'r_knee': 26, 
        'l_ankle': 27, 'r_ankle': 28
    }

    @staticmethod
    def calculate_3d_angle(a, b, c):
        if not a or not b or not c: return 0.0

        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 1)

    @classmethod
    def get_body_angles(cls, landmarks):
        angles = {}
        if not landmarks: return angles

        def calc_if_exists(p1_idx, p2_idx, p3_idx):
            if p1_idx in landmarks and p2_idx in landmarks and p3_idx in landmarks:
                return cls.calculate_3d_angle(landmarks[p1_idx], landmarks[p2_idx], landmarks[p3_idx])
            return None

        # --- Knees & Elbows ---
        r_knee = calc_if_exists(cls.KP['r_hip'], cls.KP['r_knee'], cls.KP['r_ankle'])
        if r_knee is not None: angles['right_knee'] = r_knee
        l_knee = calc_if_exists(cls.KP['l_hip'], cls.KP['l_knee'], cls.KP['l_ankle'])
        if l_knee is not None: angles['left_knee'] = l_knee

        r_elbow = calc_if_exists(cls.KP['r_shoulder'], cls.KP['r_elbow'], cls.KP['r_wrist'])
        if r_elbow is not None: angles['right_elbow'] = r_elbow
        l_elbow = calc_if_exists(cls.KP['l_shoulder'], cls.KP['l_elbow'], cls.KP['l_wrist'])
        if l_elbow is not None: angles['left_elbow'] = l_elbow

        # --- Arm to Body ---
        r_arm_body = calc_if_exists(cls.KP['r_hip'], cls.KP['r_shoulder'], cls.KP['r_elbow'])
        if r_arm_body is not None: angles['right_arm_body'] = r_arm_body
        l_arm_body = calc_if_exists(cls.KP['l_hip'], cls.KP['l_shoulder'], cls.KP['l_elbow'])
        if l_arm_body is not None: angles['left_arm_body'] = l_arm_body

        # --- Spine (FIXED) ---
        for side in ['r_shoulder', 'l_shoulder']:
            shoulder_idx = cls.KP[side]
            hip_idx = cls.KP['r_hip'] if side == 'r_shoulder' else cls.KP['l_hip']
            
            if shoulder_idx in landmarks and hip_idx in landmarks:
                shoulder = landmarks[shoulder_idx]
                hip = landmarks[hip_idx]
                # חזרנו למינוס 50 כדי שהקו הווירטואלי יצביע למעלה (היכן שהכתפיים נמצאות)
                vertical_ref = type('Point', (), {'x': hip.x, 'y': hip.y - 50.0, 'z': hip.z})()
                angles['spine'] = cls.calculate_3d_angle(shoulder, hip, vertical_ref)
                break 

        # --- Legs Spread ---
        if cls.KP['l_hip'] in landmarks and cls.KP['r_hip'] in landmarks and \
           cls.KP['l_ankle'] in landmarks and cls.KP['r_ankle'] in landmarks:
            l_hip = landmarks[cls.KP['l_hip']]
            r_hip = landmarks[cls.KP['r_hip']]
            pelvis_x = (l_hip.x + r_hip.x) / 2.0
            pelvis_y = (l_hip.y + r_hip.y) / 2.0
            pelvis_z = (l_hip.z + r_hip.z) / 2.0
            pelvis = type('Point', (), {'x': pelvis_x, 'y': pelvis_y, 'z': pelvis_z})()
            l_ankle = landmarks[cls.KP['l_ankle']]
            r_ankle = landmarks[cls.KP['r_ankle']]
            angles['legs_spread'] = cls.calculate_3d_angle(l_ankle, pelvis, r_ankle)

        return angles