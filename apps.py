from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, send_file
from flask_sqlalchemy import SQLAlchemy
import cv2
import easyocr
import re
import os
from ultralytics import YOLO
import random
from datetime import datetime
from sqlalchemy import desc
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://login:pass@host/DBname'
app.config['SECRET_KEY'] = 'your_secret_key'  # Секретный ключ для сессий
db = SQLAlchemy(app)

# Путь к директории для загруженных видео
UPLOAD_FOLDER = 'uploaded_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Проверка существования директории для загруженных видео, и создание ее, если она не существует
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists('static/uploaded_images'):
    os.makedirs('static/uploaded_images')


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Переопределение отношений в таблице Detection
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    object_name = db.Column(db.String(50), nullable=False)
    typeFire = db.Column(db.String(50), nullable=False)
    coordinates = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(200), nullable=True)  # Добавляем поле для хранения пути к изображению
    account_timestamp_id = db.Column(db.Integer, db.ForeignKey('account_timestamp.id'))

    def __repr__(self):
        return f'<Detection {self.id}>'

# Создание таблицы для сохранения текущего времени с датой и внешнего ключа id_account от таблицы account
class AccountTimestamp(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    id_account = db.Column(db.Integer, db.ForeignKey('account.id_account'))

    def __repr__(self):
        return f'<AccountTimestamp {self.id}>'

# Внешний ключ для таблицы Detection
db.ForeignKeyConstraint(['id_account'], ['account.id_account'])



# Маршруты для веб-приложения
@app.route('/')
def index():
    session.pop('video_path', None)  # Очистка переменной video_path при выходе из сессии
    session.pop('output_video_path', None)

    # Путь к папке uploaded_videos
    folder_path = os.path.join(os.getcwd(), 'uploaded_videos')

    # Очистка папки uploaded_videos
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

    return render_template('tabs.html', logged_in='username' in session)

def process_video_with_tracking(video_path, id_account, speed=1, find_coordinates=True, find_fire_type=True , show_video=True, save_video=False,
                                output_video_path="uploaded_videos\\output_video.mp4"):
    with app.app_context():
        model = YOLO('runs/detect/train3/weights/best.pt')

        reader = easyocr.Reader(['en'], gpu=True)
        # Open the input video file
        cap = cv2.VideoCapture(video_path)
        typeModel = load_model('model/model.keras')
        class_names = {0: 'низовой', 1: 'верховой'}

        # Создаем новую запись в таблице AccountTimestamp
        new_account_timestamp = AccountTimestamp(timestamp=datetime.now(), id_account=id_account)
        db.session.add(new_account_timestamp)
        db.session.commit()

        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        # Get input video frame rate and dimensions
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        names = {}
        timeID = {}
        listId = []
        # Define the output video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        results = None  # Initialize results with None
        sumTime = 0
        fullTimeStart = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get current frame number
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((150, 150))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            startTime = time.time() - fullTimeStart
            if frame_num % speed == 0:
                results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=(640, 384), verbose=False, tracker="botsort.yaml")
            endTime = time.time()
            sumTime += endTime - startTime
            # If detections occur, calculate current timestamp and add to frame
            if results is not None and len(results) > 0:
                if isinstance(video_path, int):
                    timestamp = startTime
                else:
                    timestamp = frame_num / fps
                timestamp1 = time.strftime("%H:%M:%S", time.gmtime(timestamp))
                cv2.putText(frame, f'Timestamp: {timestamp1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)


                # Here you can save the frame or timestamp if needed

            if results is not None and results[0].boxes.id is not None:
                # this will ensure that id is not None -> exist tracks
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, id, clss in zip(boxes, ids, classes):
                    # Generate a random color for each object based on its ID

                    random.seed(int(id))
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                    cv2.putText(
                        frame,
                        f"Id {id}, {model.model.names[clss]}",
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                    names.update({id:model.model.names[clss]})
                    if not id in listId:
                        listId.append(id)
                        timeID.update({id: round(timestamp, 2)})

                        # Сохраняем изображение
                        image_name = f"{model.model.names[clss]}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                        image_path = f"static/uploaded_images/{image_name}"
                        cv2.imwrite(image_path, frame)

                        coords = "Координаты отсутствуют"

                        # Извлечение координат из текущего кадра
                        if find_coordinates:
                            resCoord = reader.readtext(gray)

                            # Поиск координат в распознанном тексте
                            for (bbox, text, prob) in resCoord:
                                # Используем регулярное выражение для поиска координат (например, "lat: 12.34, lon: 56.78")
                                match = re.search(r'\s*([-+]?[0-9]*\.?[0-9]+),?\s*,\s*([-+]?[0-9]*\.?[0-9]+)', text)
                                if match:
                                    lat, lon = match.groups()
                                    coords = f"lat: {lat}, lon: {lon}"
                            if not coords:
                                coords = "Координаты отсутствуют"
                        else:
                            coords = "Координаты отсутствуют"

                        #countF = 0
                        if find_fire_type:
                            if model.model.names[clss] == 'Fire':
                                NameDetected = "Огонь"
                                predictions = typeModel.predict(img_array)
                                score = predictions[0]
                                typeFire = ("Cкорее всего {} ({:.2f}%)".format(
                                    class_names[np.argmax(score)],
                                    100 * np.max(score)))
                            else:
                                typeFire = "Отсутствует"
                                NameDetected = "Дым"
                        else:
                            typeFire = "Отсутствует"
                            if model.model.names[clss] == 'Fire':
                                NameDetected = "Огонь"
                            else:
                                NameDetected = "Дым"



                        # Создаем запись Detection с путем к изображению
                        new_detection = Detection(
                            object_name=NameDetected,
                            typeFire=typeFire,
                            coordinates=coords,
                            timestamp=round(timestamp, 2),
                            image_path=image_path,
                            account_timestamp_id=new_account_timestamp.id
                        )
                        db.session.add(new_detection)
                        db.session.commit()




            if save_video:
                out.write(frame)

            if show_video and results is not None and frame_num % speed == 0:
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (0, 0), fx=0.5, fy=0.5)
                frame = cv2.imencode('.jpg', annotated_frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            #time.sleep(0.01)  # Уменьшение задержки между кадрами

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the input video capture and output video writer
        cap.release()
        if save_video:
            out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()




@app.route('/video_feed')
def video_feed():
    if 'video_path' in session:
        video_path = session['video_path']
        speed = session.get('speed', 1)
        find_coordinates = session.get('find_coordinates', 'yes') == 'yes'
        find_fire_type = session.get('find_fire_type', 'yes') == 'yes'
        output_video_path = "uploaded_videos\\output_video.mp4"
        session['output_video_path'] = output_video_path
        # Получаем логин из сессии
        login = session.get('username')
        # Получаем id_account по логину из базы данных
        user = Account.query.filter_by(login=login).first()
        id_account = user.id_account
        return Response(process_video_with_tracking(video_path, id_account, speed, find_coordinates, find_fire_type, show_video=True, save_video=True, output_video_path="uploaded_videos\\output_video.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Обработка случая, если название видео не было сохранено в сессии
        return 'Ошибка: Название видео не найдено в сессии.'


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' in request.files:
        video_file = request.files['video_file']
        speed = request.form.get('speed', '1')
        find_coordinates = request.form.get('coordinates', 'yes')
        find_fire_type = request.form.get('fire_type', 'yes')

        # Сохраняем загруженное видео в папку на сервере
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        session['video_path'] = video_path
        session['speed'] = int(speed)
        session['find_coordinates'] = find_coordinates
        session['find_fire_type'] = find_fire_type

        return render_template('video.html', video_path=video_path)
    else:
        return 'Ошибка загрузки видео.'


@app.route('/history')
def history():
    login = session.get('username')
    user = Account.query.filter_by(login=login).first()
    id_account = user.id_account

    account_timestamps = AccountTimestamp.query.filter_by(id_account=id_account).order_by(
        AccountTimestamp.timestamp.desc()).all()
    detections_by_timestamp = {}

    for timestamp in account_timestamps:
        detections = Detection.query.filter_by(account_timestamp_id=timestamp.id).all()
        detections_by_timestamp[timestamp] = detections

    return render_template('history.html', detections_by_timestamp=detections_by_timestamp)


@app.route('/latest_history')
def latest_history():
    login = session.get('username')
    user = Account.query.filter_by(login=login).first()
    id_account = user.id_account

    latest_timestamp = AccountTimestamp.query.filter_by(id_account=id_account).order_by(AccountTimestamp.timestamp.desc()).first()

    if not latest_timestamp:
        return render_template('history.html', detections_by_timestamp={})

    detections = Detection.query.filter_by(account_timestamp_id=latest_timestamp.id).all()

    detections_by_timestamp = {latest_timestamp: detections}

    return render_template('history.html', detections_by_timestamp=detections_by_timestamp)



@app.route('/show_processed_video')
def show_processed_video():
    login = session.get('username')
    user = Account.query.filter_by(login=login).first()
    id_account = user.id_account

    latest_account_timestamp_id = AccountTimestamp.query.filter_by(id_account=id_account).order_by(
        desc(AccountTimestamp.id)).first().id
    detections = Detection.query.filter_by(account_timestamp_id=latest_account_timestamp_id).all()

    return render_template('preview.html', detections=detections)


@app.route('/live_analysis', methods=['POST'])
def live_analysis():
    video_source = request.form.get('video_source', 0)
    find_coordinates = request.form.get('coordinates', 'yes')
    find_fire_type = request.form.get('fire_type', 'yes')
    try:
        video_source = int(video_source)
    except ValueError:
        return "Invalid video source", 400

    session['video_path'] = video_source
    session['find_coordinates'] = find_coordinates
    session['find_fire_type'] = find_fire_type
    return render_template('video.html', video_path=video_source)


@app.route('/get_video_sources')
def get_video_sources():
    sources = []
    index = 0
    while index < 10:  # Устанавливаем разумный предел для индексации источников
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            sources.append(index)
            cap.release()
        index += 1
    return jsonify(sources)


@app.route('/download_video')
def download_video():
    output_video_path = session.get('output_video_path')
    if output_video_path:
        return send_file(output_video_path, as_attachment=True)
    else:
        return 'Error: No video available for download.', 404


# Определение модели для ролей
class Role(db.Model):
    id_role = db.Column(db.Integer, primary_key=True)
    role_name = db.Column(db.String(50), unique=True, nullable=False)


# Определение модели для аккаунтов
class Account(db.Model):
    id_account = db.Column(db.Integer, primary_key=True)
    id_role = db.Column(db.Integer, db.ForeignKey('role.id_role'), nullable=False)
    login = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(50), nullable=False)


# Определение модели для типов
class Types(db.Model):
    id_types = db.Column(db.Integer, primary_key=True)
    type_name = db.Column(db.String(50), unique=True, nullable=False)


# Определение модели для записей "firelist"
class Firelist(db.Model):
    firelist_id = db.Column(db.Integer, primary_key=True)
    id_types = db.Column(db.Integer, db.ForeignKey('types.id_types'), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    coordinates = db.Column(db.String(50), nullable=False)
    screenshot = db.Column(db.String(100), nullable=False)
    parent = db.Column(db.String(50), nullable=False)
    child = db.Column(db.String(50), nullable=False)
    id_account = db.Column(db.Integer, db.ForeignKey('account.id_account'), nullable=False)


# Создание всех таблиц, если они еще не существуют
with app.app_context():
    db.create_all()

    # Проверка наличия роли 'user'
    if not Role.query.filter_by(role_name='user').first():
        new_role = Role(role_name='user')
        db.session.add(new_role)
        db.session.commit()


# Регистрация нового пользователя
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        login = request.form['login']
        password = request.form['password']
        # Проверяем, существует ли уже пользователь с таким логином
        existing_user = Account.query.filter_by(login=login).first()
        if existing_user:
            flash('Пользователь с таким логином уже существует!', 'error')
        else:
            # Получаем id_role по умолчанию
            default_role = Role.query.filter_by(role_name='user').first()
            # Создаем нового пользователя с id_role по умолчанию
            new_user = Account(id_role=default_role.id_role, login=login, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('register.html')


# Вход пользователя
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login = request.form['login']
        password = request.form['password']
        # Проверяем существование пользователя и правильность введенного пароля
        user = Account.query.filter_by(login=login).first()
        if user and user.password == password:
            session['username'] = user.login
            return redirect(url_for('index'))
        else:
            flash('Неправильный логин или пароль.', 'error')
    return render_template('login.html')


# Выход пользователя
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


# Запуск приложения
if __name__ == "__main__":
    app.run(debug=False)
