import numpy as np
import tensorflow as tf

imagePath = '/Users/Jihyun/Downloads/Recycle_test/recycleproject/recycleapp/static/img/saved_img.jpg'  #절대경로, 추론을 진행할 이미지 경로
modelFullPath = '/tmp/tst/output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp/tst/output_labels.txt'                                   # 읽어들일 labels 파일 경로


def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None
    a_view=[]
    b_view=[]

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
            a_view.append(human_string)
            b_view.append(score)

        answer = labels[top_k[0]]
        #return answer
        return a_view,b_view


if __name__ == '__main__':
    run_inference_on_image()
