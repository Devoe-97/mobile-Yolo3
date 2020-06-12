import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename, save_path):
        self.cluster_number = cluster_number
        self.filename = filename
        self.save_path = save_path

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open(self.save_path, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.rstrip(" \n").split(" ")#由于每一行最后面是 “ ”+“\n”,故去掉" \n"
            length = len(infos)
            print(infos)
            for i in range(1, length):
                print(infos[i].split(","))
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)

        print("K anchors:\n {}".format(result))
        print(result)
        # list_file = open(self.save_path, 'w')
        # #list_file.write("{} {} {} ".format(result[0], result[1], result[2]))
        # list_file.write("{},{},  ".format(list(result[0])[0],list(result[0])[1]) + "{},{},  ".format(list(result[1])[0],list(result[1])[1])
        #                +"{},{},  ".format(list(result[2])[0],list(result[2])[1]) + "{},{},  ".format(list(result[3])[0],list(result[3])[1])
        #                +"{},{},  ".format(list(result[4])[0],list(result[4])[1]) + "{},{}".format(list(result[5])[0],list(result[5])[1]))
        # list_file.close()


        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 6
    filename = "MVI-20011-12.txt"
    save_path = "MVI-20011-12_tiny_yolo_anchor.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename,save_path)
    kmeans.txt2clusters()
