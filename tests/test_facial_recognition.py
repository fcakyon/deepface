import unittest
from deepface import DeepFace

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestFaceRecognition(unittest.TestCase):
    def test_bulk_face_recognition(self):
        print("Bulk face recognition tests")

        dataset = [
            ['tests/dataset/img1.jpg', 'tests/dataset/img2.jpg', True],
            ['tests/dataset/img5.jpg', 'tests/dataset/img6.jpg', True]
        ]

        resp_obj = DeepFace.verify(dataset)
        self.assertEqual(resp_obj[0]["verified"], True)
        self.assertEqual(resp_obj[1]["verified"], True)

    def test_models_and_metrics(self):
        print("Face recognition tests")

        dataset = [
            ['tests/dataset/img1.jpg', 'tests/dataset/img2.jpg', True],
            ['tests/dataset/img1.jpg', 'tests/dataset/img3.jpg', False]
        ]

        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
        metrics = ['cosine', 'euclidean', 'euclidean_l2']

        passed_tests = 0
        test_cases = 0

        for model in models:
            for metric in metrics:
                for instance in dataset:
                    img1 = instance[0]
                    img2 = instance[1]
                    result = instance[2]

                    resp_obj = DeepFace.verify(img1, img2, model_name=model, distance_metric=metric)
                    prediction = resp_obj["verified"]
                    distance = round(resp_obj["distance"], 2)
                    required_threshold = resp_obj["max_threshold_to_verify"]

                    test_result_label = "failed"
                    if prediction == result:
                        passed_tests = passed_tests + 1
                        test_result_label = "passed"

                    if prediction:
                        classified_label = "verified"
                    else:
                        classified_label = "unverified"

                    test_cases = test_cases + 1

                    print(img1, " and ", img2, " are ", classified_label, " as same person based on ", model, " model and ", metric, " distance metric. Distance: ", distance, ", Required Threshold: ", required_threshold, " (", test_result_label, ")")

                print("--------------------------")

        print("Passed unit tests: ",passed_tests," / ",test_cases)

        accuracy = 100 * passed_tests / test_cases
        accuracy = round(accuracy, 2)

        self.assertEqual(accuracy > 75, True)


if __name__ == '__main__':
    unittest.main()
