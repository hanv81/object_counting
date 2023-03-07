import cv2

class Detector:
    def __init__(self, core, model_path, thr=0.2, device='CPU'):
        self.thr = thr

        self.model = core.read_model(model_path)
        if len(self.model.inputs) != 1:
            raise RuntimeError("Detector supports only models with 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("Detector supports only models with 1 output layer")

        input_shape = self.model.inputs[0].shape
        self.nchw_layout = input_shape[1] == 3

        OUTPUT_SIZE = 7
        output_shape = self.model.outputs[0].shape
        if len(output_shape) != 4 or output_shape[3] != OUTPUT_SIZE:
            raise RuntimeError("Expected model output shape with {} outputs".format(OUTPUT_SIZE))

        compiled_model = core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()
        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if self.nchw_layout:
            _, _, self.input_h, self.input_w = input_shape
        else:
            _, self.input_h, self.input_w, _ = input_shape

    def _preprocess(self, img):
        self._h, self._w, _ = img.shape
        if self._h != self.input_h or self._w != self.input_w:
            img = cv2.resize(img, dsize=(self.input_w, self.input_h), fy=self._h / self.input_h,
                             fx=self._h / self.input_h)
        if self.nchw_layout:
            img = img.transpose(2, 0, 1)
        return img[None, ]

    def _infer(self, prep_img):
        input_data = {self.input_tensor_name: prep_img}
        output = self.infer_request.infer(input_data)[self.output_tensor]
        return output[0,0,:,1:]

    def _postprocess(self, bboxes):
        def coord_translation(bbox):
            xmin = int(self._w * bbox[2])
            ymin = int(self._h * bbox[3])
            xmax = int(self._w * bbox[4])
            ymax = int(self._h * bbox[5])
            return [xmin, ymin, xmax, ymax, bbox[1], int(bbox[0]-1)]

        return [coord_translation(bbox) for bbox in bboxes if 1 <= bbox[0] and bbox[1] > self.thr]

    def detect(self, img):
        img = self._preprocess(img)
        output = self._infer(img)
        bboxes = self._postprocess(output)
        return bboxes
