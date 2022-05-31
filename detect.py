# wenet-main/wenet/bin/recognoize_onnx.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import onnx
from onnx import helper


def onnx_new_generate():
    model = onnx.load("CAENet.onnx")
    prob_info = helper.make_tensor_value_info('45', onnx.TensorProto.FLOAT, [1, 3, 416, 416])
    model.graph.output.insert(0, prob_info)
    onnx.save(model, 'CAENet_new.onnx')


def forward(file):
    model_onnx = onnx.load("CAENet_new.onnx")
    try:
        onnx.checker.check_model(model_onnx)  # check onnx model
    except onnx.checker.ValidationError as e:
        print("model is invalid: %s" % e)

    # 读取onnx，创建session
    encoder_ort_session = rt.InferenceSession('CAENet_new.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = encoder_ort_session.get_inputs()[0].name

    # 读取图片
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    b, g, r = cv2.split(img)
    merged = cv2.merge([b, b, b, b, b, b, b, r]) / 255.0
    img_tensor_merged = np.array(merged).astype(np.float32).transpose((2, 0, 1)).reshape((1, 8, 416, 416))
    # 高维输入显示
    plt.matshow(img_tensor_merged[0, 7, :, :])

    # 输入编码器
    outputs = encoder_ort_session.run(output_names=['45'], input_feed={input_name: img_tensor_merged})

    # 降维输出
    print(outputs[0].shape)
    # 低维输出显示
    plt.matshow(outputs[0][0, 0, :, :])
    plt.show()


if __name__ == '__main__':
    file = "./testset/220516_191308_bmp.rf.6393b889caae2005a2043d102183374b.jpg"
    onnx_new_generate()
    forward(file)
    # model = onnx.load("CAENet.onnx")
    # print(model.graph)

