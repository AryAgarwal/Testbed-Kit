import streamlit as st
import riva.client
import wave
import json
import yaml
import random
import string
import secrets
import time
import base64

import riva.client.proto.riva_asr_pb2 as rasr
import numpy as np
import tritonclient.http as http_client
from tritonclient.utils import *

import librosa
import shortuuid
import soundfile as sf
import threading

import math
from kubernetes import client, config, utils
import os
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from streamlit_oauth import OAuth2Component
import os
import pickle

NAMESPACE = "test-models"
PORT_NUMBER = 8000

if 'page' not in st.session_state:
    st.session_state.page = 'table'
if 'deployed_models' not in st.session_state:
    st.session_state.deployed_models = []
if 'id' not in st.session_state:
    st.session_state.id = ""

# Initialize Kubernetes configuration
config.load_kube_config()

def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

def get_input_for_triton(audio_signal: list, src_lang: str):
    input0 = http_client.InferInput("WAV", audio_signal.shape, "FP32")
    input0.set_data_from_numpy(audio_signal)
    return [
        input0,
        get_string_tensor([[src_lang]] * len(audio_signal), "LANGUAGE"),
    ]

def load_audio(wav_path):
    audio, sr = librosa.load(wav_path, sr=16000)
    return audio, sr

def get_audio_input_for_triton(audio_filepath):
    waveform, sample_rate = load_audio(audio_filepath)
    duration = int(len(waveform) / sample_rate)
    padding_duration = 1
    samples = np.zeros(
        (1, padding_duration * sample_rate * ((duration // padding_duration) + 1)),
        dtype=np.float32,
    )
    samples[0, : len(waveform)] = waveform
    inputs = [
        http_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
    ]
    inputs[0].set_data_from_numpy(samples)
    return inputs

def get_audio_input_for_triton2(audio_signal):
    input0 = http_client.InferInput("WAV", audio_signal.shape, "FP32")
    input0.set_data_from_numpy(audio_signal)
    return [input0]

def get_translation_input_for_triton(input_text, language, task, temperature):
    temperature_obj = np.array([[temperature]], dtype=np.float32)
    temperature_input = http_client.InferInput(
            "TEMPERATURE", temperature_obj.shape, np_to_triton_dtype(temperature_obj.dtype)
        )
    temperature_input.set_data_from_numpy(temperature_obj)
    return [
        get_string_tensor([[input_text]], "INPUT_TEXT"),
        get_string_tensor([[language]], "LANGUAGE"),
        get_string_tensor([[task]], "TASK"),
        temperature_input
    ]

def batchify(arr, batch_size=1):
    num_batches = math.ceil(len(arr) / batch_size)
    return [arr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

def pad_batch(batch_data):
    batch_data_lens = np.asarray([len(data) for data in batch_data], dtype=np.int32)
    max_length = max(batch_data_lens)
    batch_size = len(batch_data)
    padded_zero_array = np.zeros((batch_size,max_length),dtype=np.float32)
    for idx, data in enumerate(batch_data):
        padded_zero_array[idx,0:batch_data_lens[idx]] = data
    return padded_zero_array, np.reshape(batch_data_lens,[-1,1])

def triton_infer(input_filepaths, model_name, input_lang, triton_http_client, prompt="", max_batch_size=1):
    raw_audio_data = [load_audio(fname)[0] for fname in input_filepaths]
    batches = batchify(raw_audio_data, batch_size=max_batch_size)
    final_translations = []
    for i in range(len(batches)):
        if "whisper" in model_name:
            inputs = get_translation_input_for_triton(prompt, "en", "translate", 0.0) + get_audio_input_for_triton2(np.array(batches[i]))
        else:
            if max_batch_size == 1:
                audio_signal = np.array(batches[i])
            else:
                audio_signal, audio_len = pad_batch(batches[i])        
            inputs = get_input_for_triton(audio_signal, input_lang)  
        output0 = http_client.InferRequestedOutput("TRANSCRIPTS")
        # print(inputs)
        response = triton_http_client.infer(
            model_name,
            model_version='1',
            inputs=inputs,
            outputs=[output0],
            headers={}, 
        )
        output_batch = response.as_numpy('TRANSCRIPTS').tolist()
        if "whisper" in model_name:
            try:
                final_translations.append(output_batch.decode("utf-8").strip())
            except:
                final_translations.append(output_batch[0].decode("utf-8").split("<|startoftranscript|><|en|><|translate|><|notimestamps|>")[-1])
        else:
            tgt_sentences = [translation.decode("utf-8") for translation in output_batch]            
            final_translations.extend(tgt_sentences)
    return final_translations

def send_triton_request(input_filepaths, model_name, input_lang, url, enable_ssl=False, prompt=""):
    print(url)
    if enable_ssl:
        import gevent.ssl
        triton_http_client = http_client.InferenceServerClient(
            url=url, verbose=False,
            ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context,
        )
    else:
        triton_http_client = http_client.InferenceServerClient(
            url=url, verbose=False,
        )
    return triton_infer(input_filepaths, model_name, input_lang, triton_http_client, prompt, 1)

def xcribe_xlate_riva(audio_file, language_code, url,hotwords=[], hotword_weight=1.0, streaming=True, task="transcribe"):
    with wave.open(audio_file, 'rb') as wav_f:
        sample_rate = wav_f.getframerate()
    server = url
    auth = riva.client.Auth(uri=server)
    asr_service_normal = riva.client.ASRService(auth)
    asr_config_offline = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=language_code,
        max_alternatives=1,
        profanity_filter=False,
        enable_automatic_punctuation=False,
        verbatim_transcripts=True,
        sample_rate_hertz=sample_rate,
        audio_channel_count=1,
    )
    if len(hotwords) > 0:
        speech_context = rasr.SpeechContext()
        speech_context.phrases.extend(hotwords)
        speech_context.boost = hotword_weight
        asr_config_offline.speech_contexts.append(speech_context)
    
    if streaming:
        asr_config_streaming = riva.client.StreamingRecognitionConfig(
            config=asr_config_offline,
            interim_results=False,
        )
        audio_generator = riva.client.AudioChunkFileIterator(audio_file, chunk_n_frames=sample_rate)
        responses = asr_service_normal.streaming_response_generator(
            audio_chunks=audio_generator, streaming_config=asr_config_streaming
        )        
        input_text = ""
        for response in responses:
            if not response.results:
                continue
            for result in response.results:
                input_text += result.alternatives[0].transcript.strip()
    else:
        audio = open(audio_file, 'rb').read()
        response = asr_service_normal.offline_recognize(audio, asr_config_offline)        
        input_text = ""
        for result in response.results:
            input_text += result.alternatives[0].transcript.strip()
    return input_text

from jinja2 import Template

def generate_2_char_id():
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(2)).lower()

def generate_service_yaml(uid):
    with open('template.yaml') as file:
        template_str = file.read()
    
    template = Template(template_str)
    # id = generate_2_char_id()
    rendered_yaml = template.render(model_name = uid, uid = uid)
    
    #  apply this YAML to your Kubernetes cluster
    return rendered_yaml,uid

def deploy_model(models,uid):
    yaml_file = "deployment-riva-combined-dev.yaml"
    # for model in models:
    try:
        # print('here')
        content,id = generate_service_yaml(uid)
        # print(content)
        with open(yaml_file, 'w') as file:
            file.write(content)
        utils.create_from_yaml(client.ApiClient(), yaml_file, namespace=NAMESPACE)
        print("deployed")
        return f"Model deployed successfully.",id
    except Exception as e:
        return f"Failed to deploy model: {str(e)}",id
    
    

def undeploy_model(uid):
    api_instance = client.AppsV1Api()
    model_name = f"riva-combined-dev-{uid}"
    try:
        api_instance.delete_namespaced_deployment(name=model_name, namespace=NAMESPACE)
        return f"Model {model_name} undeployed successfully."
    except Exception as e:
        return f"Failed to undeploy model {model_name}: {str(e)}"

def undeploy_service(uid):
    
    v1 = client.CoreV1Api()
    service_name = f"riva-combined-dev-service-{uid}"
    try:
        v1.delete_namespaced_service(name=service_name, namespace=NAMESPACE)
        return f"Service {service_name} undeployed successfully."
    except Exception as e:
        return f"Failed to undeploy service"

def scale_model_to_zero(model_name):
    api_instance = client.AppsV1Api()
    body = {'spec': {'replicas': 0}}
    try:
        api_instance.patch_namespaced_deployment_scale(name=model_name, namespace=NAMESPACE, body=body)
        return f"Model {model_name} scaled to zero."
    except Exception as e:
        return f"Failed to scale model {model_name} to zero: {str(e)}"

def get_model_to_ip(model_name, service_name):
    v1 = client.CoreV1Api()
    
    while True:
        service_to_ip = {}
        services = v1.list_namespaced_service(NAMESPACE)
        for service in services.items:
            if service.spec.type == "LoadBalancer":
                ip = service.status.load_balancer.ingress[0].ip if  service.status.load_balancer.ingress else "pending"
            elif service.spec.type in ["ClusterIP", "NodePort"]:
                ip = service.spec.cluster_ip
            else:
                ip = "N/A"
                
            service_to_ip[service.metadata.name] = f'{ip}:{PORT_NUMBER}'
            print(service.metadata.name)
        if "pending" not in service_to_ip[service_name]:
                return service_to_ip[service_name]
        print(f"Waiting for LoadBalancer IP for service {service_name}")


def transcribe_audio(audio_file, model_choices, id,language_choice, reference_text, hotwords, hotword_weight, prompt):
    sample_id = shortuuid.uuid()
    hotwords = [word.strip().lower() for word in hotwords.split(",")] if hotwords else []
    print(audio_file, model_choices,id,language_choice, reference_text, hotwords, hotword_weight, prompt)
    audio, sr = librosa.load(audio_file, sr=16000)
    # audio_path = f"audios/{language_choice}-{sample_id}-{sr}.wav"
    audio_path = audio_file
    sf.write(audio_path, audio, sr)
    
    transcriptions = {}
    for model in model_choices:
        service_name = f"riva-combined-dev-service-{id}"
        url = get_model_to_ip(model, service_name)
        if model == "asr-nemo-offline-hi-ctc":           
            transcriptions['nemo_asr_offline-ctc'] = send_triton_request([audio_path], "nemo-asr-hi", "hi-IN", url, prompt=prompt)[0]
        if model == "asr-nemo-offline-hi-rnnt":
            transcriptions['nemo_asr_offline_rnnt'] = send_triton_request([audio_path], "nemo-asr-hi-rnnt", "hi-IN", url, prompt=prompt)[0]
        if model == "asr-riva-offline-hi-greedy-testvariant":
            transcriptions['riva_asr_offline'] = xcribe_xlate_riva(audio_file=audio_path, hotwords=hotwords, hotword_weight=hotword_weight,url=url, language_code="hi", streaming=False)
        if model == "asr-riva-streaming-hi-lm-testvariant":
            transcriptions['riva_asr_streaming_lm'] = xcribe_xlate_riva(audio_file=audio_path, hotwords=hotwords, hotword_weight=hotword_weight,url=url, language_code="hi-IN", streaming=True)
        if model == "asr-riva-offline-hi-lm-testvariant":
            transcriptions['riva_asr_offline_lm'] = xcribe_xlate_riva(audio_file=audio_path, hotwords=hotwords, hotword_weight=hotword_weight, url=url,language_code="hi-IN", streaming=False)
        if model == "asr-whisper-offline-hi-fw":
            transcriptions['large_v2_whisper_fw'] = send_triton_request([audio_path], "whisper-hi-fw", "en", url, prompt=prompt)[0]
        if model == "asr-whisper-offline-hi-hf":
            transcriptions['whisper_hi_hf'] = send_triton_request([audio_path], "whisper-hi-hf", "en", url, prompt=prompt)[0]
        if model == "asr-whisper-offline-ml-hf":
            transcriptions['whisper_ml_hf'] = send_triton_request([audio_path], "whisper-ml-hf", "hi-IN", url, prompt=prompt)[0]
        if model == "asr-whisper-offline-stock-hf":
            transcriptions['whisper_stock_hf'] = send_triton_request([audio_path], "whisper-stock-hf", "en", url, prompt=prompt)[0]

    string_out = ""
    for key, value in sorted(transcriptions.items()):
        string_out += f"[{key}]\n{value}\n                      ---------------------------------------           \n"
    return string_out, sample_id

# Streamlit UI

def go_to_table():
    st.session_state.page = 'table'
    
def go_to_transcription():
    st.session_state.page = 'transcribe'

def go_to_deployment():
    st.session_state.page = 'deploy'

model_selected = dict()

def make_json(models):
    characters = string.ascii_letters + string.digits
    uid= ''.join(secrets.choice(characters) for _ in range(2)).lower()
    json_file = f"test_model_{uid}.json"
    model_map={
        "asr-riva-streaming-hi-lm-testvariant": "0.1.5",
        "asr-riva-offline-hi-lm-testvariant": "0.1.5",
        "asr-riva-offline-hi-greedy-testvariant": "0.1.6",
        "asr-nemo-offline-hi-rnnt": "0.1.5",
        "asr-nemo-offline-hi-ctc": "0.1.5",
        "asr-whisper-offline-hi-fw": "0.1.6",
        "asr-whisper-offline-hi-hf": "0.1.7",
        "asr-whisper-offline-stock-hf": "0.1.7",
        "asr-whisper-offline-ml-hf": "0.1.1"
    } 
    for model in models:
        model_selected.update({model:model_map[model]})
    with open(json_file, 'w') as f:
        json.dump(model_selected, f)
    return uid
    


def push_to_blob(local_file_path):
# Azure Storage account details
    account_url = "https://v2vh100storage.blob.core.windows.net"
    container_name = "models"
    blob_name = f"triton_model_repositories/colocated/{local_file_path}"

    # Create a BlobServiceClient
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # Create a blob client
    blob_client = container_client.get_blob_client(blob_name)

    # Upload the JSON file to the blob
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"JSON file uploaded to blob: {blob_name}")
    
    
# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to get pod names using Kubernetes API
def get_pod_names():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=NAMESPACE)
    return [pod.metadata.name for pod in pods.items]

    

# Function to get CONTAINER METADATA
def get_pod_info(pod_name):
    try:
        # Get the pod
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)

        # Extract all environment variables
        env_vars = pod.spec.containers[0].env
        all_env_vars = {env.name: env.value for env in env_vars if env.value is not None}

        # Try to read the CONTAINER_METADATA file
        try:
            exec_command = [
                '/bin/sh',
                '-c',
                'cat $CONTAINER_METADATA'
            ]
            resp = stream(v1.connect_get_namespaced_pod_exec,
                          pod_name,
                          NAMESPACE,
                          command=exec_command,
                          stderr=True, stdin=False,
                          stdout=True, tty=False)
            metadata = json.loads(resp)
        except:
            metadata = None

        return  all_env_vars['CONTAINER_METADATA']
        

    except client.exceptions.ApiException as e:
        print(f"Exception when calling CoreV1Api->read_namespaced_pod: {e}")
        return None
    

if 'pods' not in st.session_state:
    st.session_state.pods = get_pod_names()

# Use session state to keep track of which button was clicked
if 'clicked_pod' not in st.session_state:
    st.session_state.clicked_pod = None
    
import re

def extract_uid(pod_name):
    pattern = r'riva-combined-dev-(.*?)-\w+'
    match = re.search(pattern, pod_name)    
    if match:
        return match.group(1)
    else:
        return None

# Streamlit app
def table_page():
    st.title("Model Deployment Table (Currently deployed)")
    pods_to_remove=[]
    col1, col2, col3,col4 = st.columns([4, 3, 2, 1])
    col1.markdown("### **Pods**")
    col2.write("### **Models Deployed**")
    col3.write("### **Delete**")
    col4.write("### **Test**")
    for pod in st.session_state.pods:
        
        with col1:
            st.write(pod)
        
        with col2:
            mod=[]
            pod_info = get_pod_info(pod)
            values = read_json_file(pod_info)
            for m,v in values.items():
                mod.append(m)
            st.write(f"{mod}")   
        
        with col3:
            if st.button("Undeploy",key=f"undeploy_{pod}"):
                st.session_state.clicked_pod = pod
                
        with col4:
            if st.button("Test Models",key=f"test_{pod}"):
                st.session_state.model_choices = mod
                st.session_state.id = extract_uid(pod)
                print(st.session_state.model_choices)
                go_to_transcription()
                            
        # print(st.session_state.clicked_pod)
        if st.session_state.clicked_pod:
            st.write(f"Undeploying pod: {st.session_state.clicked_pod}")
            print(st.session_state.clicked_pod)
            uid = extract_uid(pod)
            print(uid)
            st.write(undeploy_model(uid))
            st.write(undeploy_service(uid))
            st.write("Undeployment complete")
            time.sleep(3)
            # Remove the undeployed pod from the list
            st.session_state.pods = [pod for pod in st.session_state.pods if pod not in st.session_state.clicked_pod]
            
            # # Reset the clicked_pod state
            st.session_state.clicked_pod = None
            
            # Rerun the app to refresh the table
            st.rerun()
        
    st.button("Go to deployment", on_click=go_to_deployment)
    if st.button("Logout"): # if user logs out
        del st.session_state["auth"]
        del st.session_state["token"]
        st.rerun()

    

def deployment_page():
    st.title("ASR Model Deployment")
    models = [
        "asr-riva-offline-hi-greedy-testvariant",
        "asr-riva-streaming-hi-lm-testvariant",
        "asr-riva-offline-hi-lm-testvariant",
        "asr-nemo-offline-hi-rnnt",
        "asr-nemo-offline-hi-ctc",
        "asr-whisper-offline-hi-fw",
        "asr-whisper-offline-hi-hf",
        "asr-whisper-offline-stock-hf",
        "asr-whisper-offline-ml-hf"
    ]
    st.header("Select Models to Deploy")
    st.session_state.model_choices = st.multiselect(
        "Choose models:",
        [model for model in models]
    )
    # print(selected_models)
    print(st.session_state.model_choices)
    models_to_deploy = st.session_state.model_choices
    # models.append(selected_models)
    current_deployed = []
    if st.button("Deploy Selected Models"):
        with st.spinner("Deploying models..."):
            uid = make_json(models_to_deploy)
            push_to_blob(f"test_model_{uid}.json")
            result, st.session_state.id= deploy_model(models_to_deploy,uid)
        st.success(result)
        st.button("Go to Transcription", on_click=go_to_transcription)
    
    # if st.button("Scale to Zero"):
    #     with st.spinner("Scaling models to zero..."):
    #         for model in st.session_state.deployed_models:
    #             scale_result = scale_model_to_zero(model)
    #             st.text(scale_result)
    #             # st.session_state.deployed_models[model] = "Scaled to Zero"

        
    st.button("Go to Model Deployments Table", on_click=go_to_table)

def transcription_page():
    st.title("Audio Transcription")
    
    st.button("Back to Deployment", on_click=go_to_deployment)
    
    language_choice = st.radio("Select Language", ["Hindi"])
    
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])
    if uploaded_file is not None:
        audio_input = st.audio(uploaded_file, format='audio/wav')
        print("audio is here")
    
    prompt = st.text_input("Enter Prompt (for Whisper-style models)")
    hotwords = st.text_input("Enter Hotwords (comma separated, for Riva-type models)")
    hotword_weight = st.slider("Hotword Weight", 0.0, 100.0, 50.0, 5.0)
    
    if st.button("Transcribe"):
        if uploaded_file is not None :
            with st.spinner("Transcribing..."):
                print("getting there")
                # Save the uploaded file temporarily
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # print(models)
                print(st.session_state.id)
                transcription, sample_id = transcribe_audio(
                    "temp_audio.wav", st.session_state.model_choices,st.session_state.id, language_choice, 
                    "", hotwords, hotword_weight, prompt
                )
                
                st.text_area("Transcription Result", transcription, height=300)
                st.text(f"Sample ID: {sample_id}")
                print("Done")
                # Clean up the temporary file
                os.remove("temp_audio.wav")
        else:
            st.warning("Please upload an audio file and ensure at least one model is deployed.")
            
    if st.button("Logout"): # if user logs out
        del st.session_state["auth"]
        del st.session_state["token"]
        st.rerun()

# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["password"] == st.secrets["password"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store password
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show input for password.
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         return False
#     elif not st.session_state["password_correct"]:
#         # Password not correct, show input + error.
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         st.error("😕 Password incorrect")
#         return False
#     else:
#         # Password correct.
#         return True



# Define these constants
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://accounts.google.com/o/oauth2/revoke"
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
# REDIRECT_URI = "https://sarvam-testbed.azurewebsites.net"
# REDIRECT_URI = "http://localhost:8501"
REDIRECT_URI = os.getenv("REDIRECT_URI")
# SCOPE = ['https://www.googleapis.com/auth/userinfo.email', 'openid']
SCOPE = 'openid profile email'

def main():
    
    # IF NOT LOGGED IN
    if "auth" not in st.session_state:
        st.title("Sarvam Internal Testbed")
        st.write("Login to test and deploy models.")

        oauth2 = OAuth2Component(
            CLIENT_ID,
            CLIENT_SECRET,
            AUTHORIZE_URL,
            TOKEN_URL,
            REFRESH_TOKEN_URL,
            REVOKE_TOKEN_URL
        )
        result = oauth2.authorize_button(
            name="Continue with Google",
            icon="https://www.google.com.tw/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            key="google",
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True,
            pkce="S256",
        )

        if result:
            # st.write(result)
            id_token = result["token"]["id_token"]
            payload = id_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            payload = json.loads(base64.b64decode(payload))
            email = payload["email"]
            st.session_state["auth"] = email
            st.session_state["token"] = result["token"]
            st.rerun()
            
    # ALREADY LOGGED IN 
    else:
        if st.session_state.page == 'table':
            table_page()
        elif st.session_state.page == 'deploy':
            deployment_page()
        elif st.session_state.page == 'transcribe':
            transcription_page()
        elif st.button("Logout"): # if user logs out
            del st.session_state["auth"]
            del st.session_state["token"]
            st.rerun()
        else:
            st.stop()  # Don't run the rest of the app.

if __name__ == "__main__":
    st.set_page_config(page_title="ASR Model Deployment and Transcription", page_icon="🎙️", layout="wide")
    main()
    
