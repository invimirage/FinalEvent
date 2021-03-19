# -*- coding: utf-8 -*-
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models
import base64
import pandas as pd
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class audio_speech_recognizer:
    _secret_id = "AKIDlJdkbExRlwueDaqjZAaomVFlDSVOuqCL"
    _secret_key = "iTefWR6XklmIfroVyQergHqAG9qIsvkO"
    _endpoint = "asr.tencentcloudapi.com"

    def __init__(self, audio_paths:list):
        self.temp_data = {'file_path': [], 'result_data': [], 'task_id': []}
        for audio in audio_paths:
            self.upload_audio(audio)
        self.temp_data = pd.DataFrame(self.temp_data)
        self.temp_data.to_csv('results.csv', mode='a+', index=False)

    def upload_audio(self, file_path):
        print('Uploading')
        try:
            audio_path = file_path
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            audio_data_b64 = base64.b64encode(audio_data).decode('utf-8')
            cred = credential.Credential(self._secret_id, self._secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = self._endpoint

            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = asr_client.AsrClient(cred, "", clientProfile)

            req = models.CreateRecTaskRequest()
            params = {
                "EngineModelType": "16k_zh_video",
                "ChannelNum": 1,
                "SpeakerDiarization": 1,
                "SpeakerNumber": 1,
                "ResTextFormat": 2,
                "SourceType": 1,
                "Data": audio_data_b64
            }
            req.from_json_string(json.dumps(params))

            resp = client.CreateRecTask(req)
            result_json = json.loads(resp.to_json_string())
            task_id = result_json['Data']['TaskId']
            self.temp_data['file_path'].append(audio_path)
            self.temp_data['task_id'].append(task_id)
            print('Upload success, now waiting for translation')
            self.get_results(int(task_id))

        except TencentCloudSDKException as err:
            print(err)

    def get_results(self, task_id):
        try:
            cred = credential.Credential(self._secret_id, self._secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = self._endpoint

            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = asr_client.AsrClient(cred, "", clientProfile)

            req = models.DescribeTaskStatusRequest()
            params = {
                "TaskId": task_id
            }
            req.from_json_string(json.dumps(params))
            seconds = 0
            while True:
                time.sleep(1)
                seconds += 1
                resp = client.DescribeTaskStatus(req)
                result_json = json.loads(resp.to_json_string())
                result_status = result_json['Data']['StatusStr']
                if result_status in ['success', 'failed']:
                    break
                print(seconds)
            result_data = result_json['Data']['ResultDetail']
            result_data_str = json.dumps(result_data, ensure_ascii=False)
            self.temp_data['result_data'].append(result_data_str)

        except TencentCloudSDKException as err:
            print(err)

if __name__ == '__main__':
    tensent_recognizer = audio_speech_recognizer(['audios/test.wav'])