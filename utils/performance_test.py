from locust import HttpUser, task, between


class MyUser(HttpUser):
    """
    You can test '_stcore/health' as a health check of streamlit server.
    """

    wait_time = between(1, 5)

    @task
    def my_task(self):
        self.client.get("http://localhost:8501/")


# TODO: find a way to perform performance testing for planning deployment spesification.
