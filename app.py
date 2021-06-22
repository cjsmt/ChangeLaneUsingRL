from flask import Flask
from flask import request
from flask_socketio import SocketIO
import pandas as pd
from train import *
import json
import logging


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

log = logging.getLogger('werkzeug')
log.disabled = True


@app.route('/initialize', methods=['POST'])
def initialize_environment():
    initData = json.loads(request.get_data(as_text=True))
    init_state_dic = pd.DataFrame({'d_pre_goal': initData['d_pre_goal'],
                                   'd_tail_goal': initData['d_tail_goal'],
                                   'd_pre_raw': initData['d_pre_raw'],
                                   'interval_pre': initData['interval_pre'],
                                   'interval_tail': initData['interval_tail'],
                                   'v_x': initData['v_x'], 'v_y': initData['v_y'],
                                   'delta_v_goal': initData['delta_v_goal'],
                                   'delta_v_raw': initData['delta_v_raw'],
                                   'delta_a_x': initData['delta_a_x'],
                                   'delta_a_y': initData['delta_a_y']}, index=[0])
    init_action_dic = pd.DataFrame({'a_x': initData['a_x'],
                                    'a_y': initData['a_y']}, index=[0])
    init_environment(init_state_dic, init_action_dic)
    return "FINISHED INITIALIZED. READY TO START."


@app.route('/start')
def start_train():
    start()
    return 'TRAIN OVER'


@app.route('/observation', methods=['POST'])
def observe():
    updateData = json.loads(request.get_data(as_text=True))
    run_step(
        v_pre_goal=updateData['v_pre_goal'], v_tail_goal=updateData['v_tail_goal'], v_pre_raw=updateData['v_pre_raw'],
        y_pre_goal=updateData['y_pre_goal'], y_tail_goal=updateData['y_tail_goal'], y_pre_raw=updateData['y_pre_raw']
    )
    return "GET THE OBSERVATION."


@app.route('/move', methods=['GET'])
def move_forward():
    x_moved, y_moved = move_step()
    return {"moveZDistance": x_moved, "moveXDistance": y_moved}


@app.route('/game_over', methods=['POST'])
def over_episode():
    overMessage = json.loads(request.get_data(as_text=True))
    if overMessage['isOver']:
        setOver()
    return "The Environment got Done."


@app.route('/check_ready', methods=['GET'])
def check_ready():
    return 'READY TO START' if env.isReady else 'NOT READY TO START'


@app.route('/check_reset', methods=['GET'])
def check_reset():
    if env.isReset:
        setFinishedReset()
        if env.isWarmUp:
            if len(rpm) + 200 >= MEMORY_WARMUP_SIZE:
                # print('WARM UP EPISODE {}. THIS IS THE FINAL WARM UP EPISODE.'.format(env.epiNum))
                return 'WARM UP EPISODE {}. THIS IS THE FINAL WARM UP EPISODE.'.format(env.epiNum)
            else:
                # print('WARM UP EPISODE {}. REPLAY MEMORY WITH {} DATA.'.format(env.epiNum, len(rpm)))
                return 'WARM UP EPISODE {}. REPLAY MEMORY WITH {} DATA.'.format(env.epiNum, len(rpm))
        else:
            if env.epiNum // 10 > 0 & env.epiNum % 10 == 0:
                if env.epiNum % 20 == 0:
                    # print('Evaluating episode:{}   Test reward:{}'.format(env.epiNum // 50, env.evalReward.get()))
                    return 'Evaluating episode:{}   Test reward:{}'.format(env.epiNum // 50, env.evalReward.get())
                # print("Training episode:{}  Total reward:{}".format(env.epiNum, env.trainReward.get()))
                return "Training episode:{}  Total reward:{}".format(env.epiNum, env.trainReward.get())
            # print('Continue')
            return 'Continue'
    else:
        # print('NOT RESET THE ENVIRONMENT')
        return 'NOT RESET THE ENVIRONMENT'


@app.route('/finish_reset', methods=['GET'])
def finish_rest():
    setFinishedReset()
    return 'FINISHED RESET'


@socketio.on('connect')
def test_connect():
    print('connect successfully.')
    socketio.emit('my_response', {'data': '链接成功'})


@socketio.on('ready message')
def send_ready_message():
    # socketio.emit('ready_message', 'READY TO START')
    socketio.send('READY TO START')


@socketio.on('reset message')
def send_reset_message():
    socketio.send('reset_message', 'RESET THE ENVIRONMENT')


@socketio.on('move message')
def send_move_message(moved_data):
    socketio.emit('move_message', moved_data)


if __name__ == '__main__':
    app.run()
    # socketio.run(app)
