import unittest
from deepface import DeepFace

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestFacialAnalysis(unittest.TestCase):
    def test_bulk_facial_analysis(self):
        print("Bulk facial analysis tests")

        dataset = [
            'tests/dataset/img1.jpg',
            'tests/dataset/img2.jpg',
            'tests/dataset/img5.jpg',
            'tests/dataset/img6.jpg'
        ]

        resp_obj = DeepFace.analyze(dataset)
        self.assertEqual((25 < resp_obj[0]["age"] and resp_obj[0]["age"] < 45), True)
        self.assertEqual(resp_obj[0]["dominant_emotion"], "happy")
        self.assertEqual(resp_obj[0]["gender"], "Woman")

        self.assertEqual((25 < resp_obj[1]["age"] and resp_obj[1]["age"] < 45), True)
        self.assertEqual(resp_obj[1]["dominant_emotion"], "happy")
        self.assertEqual(resp_obj[1]["gender"], "Woman")

        self.assertEqual((25 < resp_obj[2]["age"] and resp_obj[2]["age"] < 45), True)
        self.assertEqual(resp_obj[2]["dominant_emotion"], "neutral")
        self.assertEqual(resp_obj[2]["gender"], "Woman")

        self.assertEqual((25 < resp_obj[3]["age"] and resp_obj[3]["age"] < 45), True)
        self.assertEqual(resp_obj[3]["dominant_emotion"], "neutral")
        self.assertEqual(resp_obj[3]["gender"], "Woman")

    def test_single_facial_analysis(self):
        print("Facial analysis test. Passing nothing as an action")

        img = "tests/dataset/img4.jpg"
        demography = DeepFace.analyze(img)
        self.assertEqual(demography["dominant_emotion"], "happy")
        self.assertEqual(demography["dominant_race"], "white")
        self.assertEqual(demography["gender"], "Woman")

        print("Facial analysis test. Passing all to the action")

        demography = DeepFace.analyze(img, ['age', 'gender', 'race', 'emotion'])
        self.assertEqual(demography["dominant_emotion"], "happy")
        self.assertEqual(demography["dominant_race"], "white")
        self.assertEqual(demography["gender"], "Woman")


if __name__ == '__main__':
    unittest.main()
