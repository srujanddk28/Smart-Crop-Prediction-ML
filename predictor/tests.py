from django.test import TestCase


class PredictorPageTests(TestCase):
    def test_homepage_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Find a crop recommendation")

    def test_dashboard_loads_without_trained_model(self):
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No results yet")

    def test_invalid_input_shows_validation_error(self):
        response = self.client.post("/predict/", {
            "N": "not-a-number",
            "P": "42",
            "K": "43",
            "temperature": "20",
            "humidity": "80",
            "ph": "6.5",
            "rainfall": "200",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter a valid number for N")
