import numpy as np

class ObjectTracker:
    def __init__(self):
        self.tracks = {} # {id: {'bbox': [x,y,w,h], 'missed': 0}}
        self.next_id = 0

    def update(self, detections):
        # detections: [[x,y,w,h], ...]
        active_ids = []

        for det in detections:
            cx, cy = det[0]+det[2]/2, det[1]+det[3]/2

            # Simple Euclidean Matching
            best_id = -1
            min_dist = 100 # pixels threshold

            for tid, data in self.tracks.items():
                lb = data['bbox']
                lcx, lcy = lb[0]+lb[2]/2, lb[1]+lb[3]/2
                dist = np.hypot(cx-lcx, cy-lcy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id != -1:
                self.tracks[best_id]['bbox'] = det
                self.tracks[best_id]['missed'] = 0
                active_ids.append({'id': best_id, 'center': (cx, cy)})
            else:
                self.tracks[self.next_id] = {'bbox': det, 'missed': 0}
                active_ids.append({'id': self.next_id, 'center': (cx, cy)})
                self.next_id += 1

        # Clean up dead tracks
        for tid in list(self.tracks.keys()):
            found = any(x['id'] == tid for x in active_ids)
            if not found:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > 5:
                    del self.tracks[tid]

        return active_ids