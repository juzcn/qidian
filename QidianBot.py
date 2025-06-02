from dotenv import load_dotenv

load_dotenv()
import logging

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from models.models_config import ModelsConfig
import os
from pathlib import Path
import json
import filetype
from importlib_resources import files
from utils.decorators import function_logging
from utils.helpers import file_to_base64, sleep_forever
from models.llm.nateshmbhat import tts
from models.llm.zhipu import glm_asr
from models.modelbase import model_stream, model_invoke
from models.llm_templates import template_to_content, TEMPLATES

import gradio as gr

share_url = ""

logger.info("Starting...")
json_content = files('resources').joinpath("gradio-config.json").read_text(encoding='utf-8')
GradioConfig = json.loads(json_content)


def prompt_textbox_attrs(model):
    sources = ["microphone"]
    if model["input_types"]:
        sources.append("upload")
        placeholder = "输入文本或语音输入,上传媒体文件"
    else:
        placeholder = "输入文本或语音输入"
    file_types = model["input_types"]
    if "audio" not in model["input_types"]:
        file_types = file_types.append("audio")
    return {"sources": sources, "file_count": "multiple", "file_types": file_types, "placeholder": placeholder}


def fn_update_model_argument(model_dropdown, model_args, argument_state, argument_component):
    #    logger.debug(f"arguments before{model_args}")
    #    logger.debug(f"to change{argument_state}={argument_component}")
    model_args[model_dropdown][argument_state[0]][argument_state[1]] = argument_component
    return model_args


def fn_update_prompt_textbox(template_dropdown, argument_state, template_variable_textbox):
    TEMPLATES[template_dropdown]['default'][argument_state] = template_variable_textbox
    return template_to_content(template_dropdown, TEMPLATES[template_dropdown]['default'])


def fn_template_change(template_dropdown):
    return template_to_content(template_dropdown), TEMPLATES[template_dropdown]['description']


def fn_update_llm_arguments(model_dropdown, model_args, argument_state, llm_dropdown, llm_model, llm_temperature):
    model_args[model_dropdown][argument_state[0]][argument_state[1][0]] = llm_dropdown
    model_args[model_dropdown][argument_state[0]][argument_state[1][1]] = llm_model
    model_args[model_dropdown][argument_state[0]][argument_state[1][2]] = llm_temperature
    return model_args


def render_llm_arguments(model_dropdown_value, argument_type, key, value):
    llm_dropdown = gr.Dropdown(value=value,
                               choices=ModelsConfig[model_dropdown_value]['info_args'][key],
                               label=key,
                               interactive=True,
                               )

    llm_model = gr.Dropdown(value=ModelsConfig[value]['model_args']['class_args']['model'],
                            choices=ModelsConfig[value]['info_args']['model'],
                            label="model",
                            interactive=True,
                            )
    llm_temperature = gr.Number(value=ModelsConfig[value]['model_args']['class_args']['temperature'],
                                label="temperature",
                                interactive=True,
                                )

    argument_state = gr.State((argument_type, [key, "model", "temperature"]))
    llm_dropdown.select(fn_update_llm_arguments,
                        [model_dropdown, model_args, argument_state, llm_dropdown, llm_model, llm_temperature],
                        [model_args])
    llm_model.select(fn_update_llm_arguments,
                     [model_dropdown, model_args, argument_state, llm_dropdown, llm_model, llm_temperature],
                     [model_args])
    llm_temperature.change(fn_update_llm_arguments,
                           [model_dropdown, model_args, argument_state, llm_dropdown, llm_model, llm_temperature],
                           [model_args])


def render_model_argument(model_dropdown_value, argument_type, key, value):
    if key == "llm":
        render_llm_arguments(model_dropdown_value, argument_type, key, value)
        return
    if "llm" in model_args.value[model_dropdown_value]["class_args"] and (key == "model" or key == "temperature"):
        return
    if isinstance(value, str):
        if key in ModelsConfig[model_dropdown_value]['info_args']:
            argument_component = gr.Dropdown(value=value,
                                             choices=
                                             ModelsConfig[model_dropdown_value]['info_args'][key],
                                             label=key,
                                             interactive=True,
                                             )
        else:
            argument_component = gr.Textbox(value=value,
                                            label=key,
                                            interactive=True,
                                            )

    else:
        argument_component = gr.Number(value=value,
                                       label=key,
                                       interactive=True,
                                       )
    argument_state = gr.State((argument_type, key))
    if isinstance(value, str):
        if key in ModelsConfig[model_dropdown_value]['info_args']:
            argument_event = argument_component.select
        else:
            argument_event = argument_component.blur

    else:
        argument_event = argument_component.change
    argument_event(fn_update_model_argument,
                   [model_dropdown, model_args, argument_state, argument_component],
                   [model_args])


def fn_app_selected(app_dropdown):
    #    [app_dropdown, model_dropdown],
    #    [model_dropdown, model_description_textbox, chatbot, prompt_textbox, chat_messages]
    chatbot_label = "多轮会话" if GradioConfig[app_dropdown]['chat'] else "单轮会话"
    model_choices = GradioConfig[app_dropdown]['model_choices']
    model = model_choices[0]
    #    print(model_choices, model, LLM_MODELS[model]['description'])
    return (
        gr.update(choices=model_choices, value=model), gr.update(value=ModelsConfig[model]['description']),
        gr.update(value=[], label=chatbot_label), gr.update(
        **prompt_textbox_attrs(GradioConfig[app_dropdown])), [],
    )


def fn_model_selected(model_dropdown):
    #    [model_dropdown, model_dropdown],
    #    [model_description_textbox, chatbot, chat_messages]
    return (
        ModelsConfig[model_dropdown]['description'], gr.update(value=[]), [])


@function_logging(handle_errors=False)
def fn_prompt_submit(app_dropdown, model_dropdown, prompt_textbox, chatbot, chat_messages, model_args):
    #    logger.debug(f"app_selected: {app_dropdown}, model selected: {model_dropdown},mm_textbox: {prompt_textbox}")
    #    logger.debug(f"app data: {GRADIO_CONFIG[app_dropdown]}")
    #    logger.debug(f"model_args: {model_args[model_dropdown]}")
    app_selected = GradioConfig[app_dropdown]
    voice_input = False
    # 1 if using microphone, convert audio input to msg[text]
    #    if app_selected['speech_to_text']:
    if prompt_textbox['files']:
        if filetype.is_audio(prompt_textbox['files'][-1]) and Path(prompt_textbox['files'][-1]).name == 'audio.wav':
            prompt_textbox['text'] = glm_asr(prompt_textbox['files'][-1])
            chatbot.append({"role": "user", "content": gr.Audio(value=prompt_textbox['files'][-1])})
            prompt_textbox['files'].pop()
            voice_input = True
    # 2 update gradio UI before model call
    chatbot.append({"role": "user", "content": prompt_textbox['text']})
    if prompt_textbox['files']:
        for file in prompt_textbox['files']:
            if filetype.is_image(file):
                chatbot.append({"role": "user", "content": gr.Image(value=file)})
            elif filetype.is_video(file):
                chatbot.append({"role": "user", "content": gr.Video(value=file)})
            elif filetype.is_audio(file):
                chatbot.append({"role": "user", "content": gr.Audio(value=file)})
    chatbot.append({"role": "assistant", "content": "", "metadata": {"title": f'{app_dropdown}（{model_dropdown}）'}})
    yield "", chatbot, chat_messages

    # 3 convert mm_textbox to model prompt
    if app_selected['chat']:
        # model chat messages
        if app_selected['message_type'] == "text":
            content = prompt_textbox['text']
        else:
            content = [{'text': prompt_textbox['text'], 'type': 'text'}]

        if prompt_textbox['files']:
            url_text = "根据提供的文件URL："
            if app_selected['message_type'] == "text":
                content = "根据提供的文件URL："
                for file in prompt_textbox['files']:
                    if filetype.is_image(file):
                        url_text += f'图片{share_url}/gradio_api/file={file}，'
                    elif filetype.is_video(file):
                        url_text += f'视频{share_url}/gradio_api/file={file}，'
                    else:
                        url_text += f'{share_url}/gradio_api/file={file}，'
                content = url_text + content
            else:  # modal message
                for file in prompt_textbox['files']:
                    if filetype.is_image(file):
                        if share_url:
                            image_url = share_url + f"/gradio_api/file={file}"
                            #                        image_url = image_url.replace('//', '/')
                            content.append(
                                {'image_url': {'url': image_url}, 'type': 'image_url'})
                        else:
                            content.append({'image_url': {'url': file_to_base64(file)}, 'type': 'image_url'})
                    #                    content[0]['text'] = f'给定[图片URL=]({share_url}/gradio_api/file={file})，' + content[0]['text']
                    elif filetype.is_video(file):
                        if share_url:
                            video_url = share_url + f"/gradio_api/file={file}"
                            #                       video_url = video_url.replace('//', '/')
                            content.append(
                                {'video_url': {'url': video_url}, 'type': 'video_url'})
                        else:
                            content.append({'video_url': {'url': file_to_base64(file)}, 'type': 'video_url'})
        chat_messages.append({'role': 'user', 'content': content})
        prompt = chat_messages
    else:
        # support only text prompt model, change in future
        prompt = prompt_textbox['text']
    #    logger.debug(f"prompt {prompt}")

    # 4 call model, streaming gradio components
    if app_selected['stream'] and ModelsConfig[model_dropdown][
        'stream_get'] is not None:  # only for model return text actually！！！
        try:
            #            logger.debug(f"Stream Call arguments : {prompt} {model_dropdown} {model_args[model_dropdown]}")
            for chunk in model_stream(prompt, ModelsConfig[model_dropdown], model_args[model_dropdown]):
                #                logger.debug(chunk)
                chatbot[-1]['content'] += chunk
                yield "", chatbot, chat_messages
        except Exception as e:
            logger.error(e)
            raise gr.Error(str(e))
        finally:
            chat_messages.append({"role": "assistant", "content": chatbot[-1]['content']})
            if voice_input:
                file = tts(chatbot[-1]['content'])
                chatbot.append({"role": "assistant", "content": gr.Audio(value=file)})
                yield "", chatbot, chat_messages
    else:
        try:
            response = model_invoke(prompt, ModelsConfig[model_dropdown], model_args[model_dropdown])
            #            logger.debug(response)
            if app_selected['output_type'] == 'text':
                #                print(f"XXXXX {chatbot} {str(response)}")
                chatbot[-1]['content'] += str(response)
                if app_selected['chat']:
                    chat_messages.append({"role": "assistant", "content": chatbot[-1]['content']})
                if voice_input:
                    file = tts(chatbot[-1]['content'])
                    chatbot.append({"role": "assistant", "content": gr.Audio(value=file)})
            else:  # image
                chatbot[-1]['content'] = gr.Image(value=response)

            yield "", chatbot, chat_messages
        except Exception as e:
            logger.error(e)
            raise gr.Error(str(e))


with (gr.Blocks(title="奇点AI演示平台",
                theme=gr.themes.Default(spacing_size="sm", radius_size="none", )) as demo):
    logger.debug(f"Gradio temp path: {os.getenv("GRADIO_TEMP_DIR")}`")
    chat_messages = gr.State([])

    model_args = {}
    for model, metadata in ModelsConfig.items():
        model_args[model] = metadata["model_args"]
    model_args = gr.State(model_args)
    with gr.Row():
        app_dropdown = gr.Dropdown(
            choices=list(GradioConfig.keys()),
            multiselect=False,
            value=list(GradioConfig.keys())[0],
            show_label=False,
            interactive=True,
            container=False,
            scale=0
        )
        model_dropdown = gr.Dropdown(
            choices=GradioConfig[app_dropdown.value]['model_choices'],
            multiselect=False,
            value=GradioConfig[app_dropdown.value]['model'],
            show_label=False,
            interactive=True,
            container=False,
            scale=0
        )
        model_description_textbox = gr.Markdown(
            #            container=False,
            value=ModelsConfig[model_dropdown.value]['description'],
            show_label=False,
        )
    chatbot_label = "多轮会话" if GradioConfig[app_dropdown.value]['chat'] else "单轮会话"
    chatbot = gr.Chatbot(
        avatar_images=("resources/assets/human.jpg", "resources/assets/robot.jpg"),
        label=chatbot_label,
        type="messages")

    prompt_textbox = gr.MultimodalTextbox(
        **prompt_textbox_attrs(GradioConfig[app_dropdown.value]),
        value=template_to_content(GradioConfig[app_dropdown.value]['prompt_template']),
        label="提示词", show_label=True, autofocus=True)
    with gr.Sidebar(position="right", open=False):
        gr.Markdown("模型参数")


        @gr.render(inputs=[model_dropdown, model_args])
        def change_model_args(model_dropdown_value, model_args_value):
            for key, value in model_args_value[model_dropdown_value]['class_args'].items():
                render_model_argument(model_dropdown_value, 'class_args', key, value)
            for key, value in model_args_value[model_dropdown_value]['call_args'].items():
                render_model_argument(model_dropdown_value, 'call_args', key, value)


        gr.Markdown("提示词模板选择")
        choices = list(TEMPLATES.keys())
        template_name = GradioConfig[app_dropdown.value]['prompt_template']
        template_dropdown = gr.Dropdown(value=template_name, choices=choices, scale=0, interactive=True,
                                        show_label=False, container=False)
        description_textbox = gr.Textbox(value=TEMPLATES[template_name]['description'], interactive=False,
                                         show_label=False, container=False)


        @gr.render(inputs=[template_dropdown])
        def load_template(template_dropdown_value):
            template_fields = TEMPLATES[template_dropdown_value]['default']
            for key, value in template_fields.items():
                template_variable_textbox = gr.Textbox(value=value,
                                                       label=key,
                                                       interactive=True,
                                                       )
                argument_state = gr.State(key)
                template_variable_textbox.change(fn_update_prompt_textbox,
                                                 [template_dropdown, argument_state,
                                                  template_variable_textbox], [prompt_textbox])

    app_dropdown.select(fn_app_selected, [app_dropdown],
                        [model_dropdown, model_description_textbox, chatbot, prompt_textbox, chat_messages, ])
    model_dropdown.select(fn_model_selected, [model_dropdown],
                          [model_description_textbox, chatbot, chat_messages])
    #    clear = gr.ClearButton([mm_textbox, chatbot, chat_messages], value="清空输入输出")
    prompt_textbox.submit(fn_prompt_submit,
                          [app_dropdown, model_dropdown, prompt_textbox, chatbot, chat_messages, model_args],
                          [prompt_textbox, chatbot, chat_messages])

    template_dropdown.change(fn_template_change, [template_dropdown], [prompt_textbox, description_textbox])

# gr.set_static_paths(paths=["./resources/assets", "./chroma_db", os.getenv("GRADIO_TEMP_DIR")])

if __name__ == "__main__":
    _, _, share_url = demo.launch(share=True, prevent_thread_lock=True, favicon_path="./resources/assets/deepseek.png",
                                  allowed_paths=["./resources/assets"])
    #    _, _, share_url = demo.launch(share=True, prevent_thread_lock=True, favicon_path="gradio_assets/deepseek.png",
    #                                  allowed_paths=["./gradio_assets", "./chroma_db", os.getenv("HF_HOME"),
    #                                                 os.getenv("GRADIO_TEMP_DIR")])
    # Keep the script running
    sleep_forever()
