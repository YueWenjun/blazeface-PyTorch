import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
plt.ion()
plt.switch_backend('agg')

def show_landmarks(image,faces):
    plt.imshow(image)
    for face in faces:
        plt.scatter(face["bbox"][0],face["bbox"][1])
        plt.scatter(face["bbox"][2],face["bbox"][3])
        plt.pause(0.001)

def parse_rec(xml_filename):
    """ Parse my label file in xml format """
    tree = ET.parse(xml_filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text) - 1,
            int(bbox.find('ymin').text) - 1,
            int(bbox.find('xmax').text) - 1,
            int(bbox.find('ymax').text) - 1,
        ]
        objects.append(obj_struct)

    return objects

plt.figure()
show_landmarks(plt.imread("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg"),parse_rec("/home/danale/disk/ywj/data/VOCdevkit/VOC2007/Annotations/000005.xml"))
plt.show()