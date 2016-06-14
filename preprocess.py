def label_to_vector(label):
    if label == "german shepherd":
        return [0, 0, 0, 0, 1]
    elif label == "anatolian shepherd":
        return [0, 0, 0, 1, 0]
    elif label == "australian shepherd":
        return [0, 0, 1, 0, 0]
    elif label == "belgian shepherd":
        return [0, 1, 0, 0, 0]
    elif label == "caucasian shepherd":
        return [1, 0, 0, 0, 0]

def vector_to_label(label):
    if label == [0, 0, 0, 0, 1]:
        return  "german shepherd"
    elif label == [0, 0, 0, 1, 0]:
        return  "anatolian shepherd"
    elif label == [0, 0, 1, 0, 0]:
        return  "australian shepherd"
    elif label == [0, 1, 0, 0, 0]:
        return  "belgian shepherd"
    elif label == [1, 0, 0, 0, 0]:
        return  "caucasian shepherd"

class DataPreprocessor:
    """
        This class provides an intuitive interface to preprocess a .csv file.
    """
    
    def __init__(self, data):
        self.data = data
    
    def del_header(self):
        return self.data[1:]
    
    def normalize(self, seq):
        def min_max(elem):
            return (max(seq) - elem) / (max(seq) - min(seq))
        return min_max
    
    def decompose(self, block):
        return tuple(block)
    
    def clear(self):
        # m - male, f - female
        # w - weight, h - height    
        m_w_list = list();        f_w_list = list()
        m_h_list = list();        f_h_list = list()
        labels = list()
        
        for row in self.del_header():
            
            m_w, f_w, m_h, f_h, br = self.decompose(row)
            
            m_w_list.append(float(m_w))
            f_w_list.append(float(f_w))
            m_h_list.append(float(m_h))
            f_h_list.append(float(f_h))
            
            labels.append(label_to_vector(br))
            
        norm_feature_iter1 = map(self.normalize(m_w_list), m_w_list)
        norm_feature_iter2 = map(self.normalize(f_w_list), f_w_list)
        norm_feature_iter3 = map(self.normalize(m_h_list), m_h_list)
        norm_feature_iter4 = map(self.normalize(f_h_list), f_h_list)
        
        features = list(zip(norm_feature_iter1,
                            norm_feature_iter2,
                            norm_feature_iter3,
                            norm_feature_iter4))
        
        return features, labels
