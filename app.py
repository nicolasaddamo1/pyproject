import cv2
import numpy as np
import time
from collections import defaultdict

class PersonCounter:
    def __init__(self, source_video, line_position=0.6, detection_mode="upper_body"):
        """
        Inicializa el contador de personas usando OpenCV.
        
        Args:
            source_video: Ruta al archivo de video o índice de la cámara.
            line_position: Posición de la línea de conteo (entre 0 y 1, por defecto 0.6 - 60% de la altura).
            detection_mode: Modo de detección: "full_body", "upper_body", o "both".
        """
        self.source = source_video
        self.line_position = line_position
        self.detection_mode = detection_mode
        
        # Detector de personas HOG+SVM de OpenCV
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detector de cuerpo superior (Haar Cascade)
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        
        # Contadores
        self.current_id = 0
        self.people_in = 0
        self.people_out = 0
        
        # Estructura para almacenar información de seguimiento
        self.tracked_objects = {}
        
        # Posiciones previas para cada ID
        self.positions = defaultdict(list)
        
        # Estado de cada persona (encima o debajo de la línea)
        self.person_states = {}
        
        # Contador de frames
        self.frame_count = 0
    
    def _detect_people(self, frame):
        """Detectar personas en el frame según el modo seleccionado."""
        # Reducir el tamaño del frame para acelerar la detección
        scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        # Lista para almacenar todas las detecciones
        all_detections = []
        
        # Convertir a escala de grises para la detección Haar Cascade
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar ecualización de histograma para mejorar el contraste
        gray = cv2.equalizeHist(gray)
        
        # Detección según el modo seleccionado
        if self.detection_mode in ["full_body", "both"]:
            # Detección de cuerpo completo con HOG+SVM
            full_body_boxes, weights = self.hog.detectMultiScale(
                small_frame, 
                winStride=(8, 8),
                padding=(4, 4), 
                scale=1.05
            )
            
            # Escalar las cajas al tamaño original y filtrar
            if len(full_body_boxes) > 0:
                full_body_boxes = np.array([[int(x/scale), int(y/scale), int(w/scale), int(h/scale)] 
                                           for (x, y, w, h) in full_body_boxes])
                
                # Filtrar por confianza y tamaño
                for i, box in enumerate(full_body_boxes):
                    x, y, w, h = box
                    if weights[i] > 0.3 and w > 50 and h > 100:
                        all_detections.append((box, "full"))
        
        if self.detection_mode in ["upper_body", "both"]:
            # Detección de cuerpo superior con Haar Cascade
            upper_body_boxes = self.upper_body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Escalar las cajas al tamaño original
            upper_body_boxes = np.array([[int(x/scale), int(y/scale), int(w/scale), int(h/scale)] 
                                       for (x, y, w, h) in upper_body_boxes])
            
            # Filtrar por tamaño
            for box in upper_body_boxes:
                x, y, w, h = box
                if w > 40 and h > 40:  # Ajustar umbrales según necesidad
                    # Ajustar la altura para simular una detección de cuerpo completo
                    adjusted_h = int(h * 2.5)  # Estimamos que el torso es ~40% del cuerpo
                    adjusted_box = [x, y, w, adjusted_h]
                    all_detections.append((adjusted_box, "upper"))
        
        # Extraer solo las cajas de detección
        boxes = [box for box, _ in all_detections]
        
        return boxes
    
    def _draw_line(self, frame):
        """Dibujar la línea de conteo en el frame."""
        height, width = frame.shape[:2]
        line_y = int(height * self.line_position)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        return line_y
    
    def _check_line_crossing(self, person_id, y_position, line_y):
        """
        Verificar si una persona ha cruzado la línea y en qué dirección.
        Retorna: -1 (salió), 1 (entró), 0 (no cruzó)
        """
        # Obtener historial de posiciones
        if person_id in self.positions:
            position_history = self.positions[person_id]
            if len(position_history) >= 3:  # Necesitamos suficiente historial
                # Obtener posiciones previas
                prev_positions = position_history[-3:]
                avg_prev_y = sum(y for y in prev_positions) / len(prev_positions)
                
                # Si no está establecido el estado, lo inicializamos
                if person_id not in self.person_states:
                    if avg_prev_y < line_y:
                        self.person_states[person_id] = "above"
                    else:
                        self.person_states[person_id] = "below"
                        
                # Verificar cruce de línea
                if self.person_states[person_id] == "above" and y_position > line_y:
                    self.person_states[person_id] = "below"
                    return 1  # Entró
                elif self.person_states[person_id] == "below" and y_position < line_y:
                    self.person_states[person_id] = "above"
                    return -1  # Salió
                    
        return 0  # No cruzó
    
    def _non_max_suppression(self, boxes, overlapThresh=0.4):
        """
        Aplica supresión no máxima para filtrar detecciones redundantes.
        """
        # Si no hay cajas, devolver array vacío
        if len(boxes) == 0:
            return []
            
        boxes = np.array(boxes)
        
        # Coordenadas de las cajas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Área de cada caja
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Índices en orden de confianza (usamos el área como proxy)
        idxs = np.argsort(area)
        
        # Lista para mantener los índices a conservar
        pick = []
        
        # Mientras queden índices para procesar
        while len(idxs) > 0:
            # Obtener el último índice (mayor área)
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Encontrar las coordenadas máximas para el inicio de las cajas
            # y las coordenadas mínimas para el final de las cajas
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # Calcular ancho y alto de la intersección
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Calcular ratio de intersección sobre unión
            overlap = (w * h) / area[idxs[:last]]
            
            # Eliminar índices con solapamiento mayor que el umbral
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        # Devolver solo las cajas seleccionadas
        return [boxes[i] for i in pick]
    
    def _match_detections_to_trackers(self, detections, tracked_objects, threshold=30):
        """
        Asocia detecciones actuales con objetos existentes en seguimiento.
        Retorna coincidencias y objetos no coincidentes.
        """
        if len(tracked_objects) == 0:
            return [], detections
        
        matches = []
        unmatched_detections = detections.copy()
        
        # Para cada objeto en seguimiento
        for obj_id, obj_data in tracked_objects.items():
            if not unmatched_detections:  # Si no quedan detecciones sin coincidencias
                break
                
            obj_center = (obj_data['x'] + obj_data['w']//2, obj_data['y'] + obj_data['h']//2)
            
            best_match_idx = -1
            min_distance = float('inf')
            
            # Buscar la detección más cercana
            for i, det in enumerate(unmatched_detections):
                x, y, w, h = det
                det_center = (x + w//2, y + h//2)
                
                distance = np.sqrt((obj_center[0] - det_center[0])**2 + (obj_center[1] - det_center[1])**2)
                
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    best_match_idx = i
            
            # Si encontramos una coincidencia
            if best_match_idx != -1:
                matches.append((obj_id, unmatched_detections[best_match_idx]))
                unmatched_detections.pop(best_match_idx)
        
        return matches, unmatched_detections
    
    def process_video(self):
        """Procesar el video e identificar personas que cruzan la línea."""
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la fuente de video.")
            return
        
        # Obtener propiedades del video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Opcionales: guardar el resultado
        # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
        
        self.frame_count = 0
        detection_interval = 10  # Detectar personas cada 10 frames (más frecuente)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Dibujar la línea de conteo
            line_y = self._draw_line(frame)
            
            # Detectar nuevas personas periódicamente
            if self.frame_count % detection_interval == 0 or len(self.tracked_objects) == 0:
                # Detectar personas
                detected_boxes = self._detect_people(frame)
                
                # Aplicar supresión no máxima para filtrar detecciones redundantes
                detected_boxes = self._non_max_suppression(detected_boxes, overlapThresh=0.5)
                
                # Asociar detecciones con objetos en seguimiento
                matches, unmatched_detections = self._match_detections_to_trackers(detected_boxes, self.tracked_objects, threshold=50)
                
                # Actualizar posiciones de objetos coincidentes
                for obj_id, det in matches:
                    x, y, w, h = det
                    self.tracked_objects[obj_id] = {'x': x, 'y': y, 'w': w, 'h': h, 'frames_since_seen': 0}
                
                # Agregar nuevos objetos para detecciones sin coincidencias
                for det in unmatched_detections:
                    x, y, w, h = det
                    self.current_id += 1
                    self.tracked_objects[self.current_id] = {'x': x, 'y': y, 'w': w, 'h': h, 'frames_since_seen': 0}
                    # Para cuerpo superior, usamos el centro inferior como punto de referencia
                    self.positions[self.current_id] = [y + h]
            
            # Incrementar contador de frames sin ver para todos los objetos
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['frames_since_seen'] += 1
                
                # Eliminar objetos que no se han visto en mucho tiempo
                if self.tracked_objects[obj_id]['frames_since_seen'] > 30:
                    del self.tracked_objects[obj_id]
                    if obj_id in self.positions:
                        del self.positions[obj_id]
                    if obj_id in self.person_states:
                        del self.person_states[obj_id]
            
            # Procesar cada persona rastreada
            for person_id, obj_data in self.tracked_objects.items():
                x, y, w, h = obj_data['x'], obj_data['y'], obj_data['w'], obj_data['h']
                
                # Dibujar el rectángulo de la persona
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calcular el punto de referencia para verificar cruces de línea
                # Para cuerpo superior, usamos la parte inferior del rectángulo
                reference_y = y + h
                
                # Actualizar historial de posiciones
                self.positions[person_id].append(reference_y)
                if len(self.positions[person_id]) > 10:  # Mantener historial limitado
                    self.positions[person_id].pop(0)
                
                # Verificar si cruzó la línea
                crossing = self._check_line_crossing(person_id, reference_y, line_y)
                if crossing == 1:
                    self.people_in += 1
                    # Resaltar la detección de entrada
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, "ENTRADA", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif crossing == -1:
                    self.people_out += 1
                    # Resaltar la detección de salida
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(frame, "SALIDA", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Mostrar ID en el frame
                cv2.putText(frame, f"ID: {person_id}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Mostrar contadores en el frame
            cv2.putText(frame, f"Personas que subieron: {self.people_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Personas que bajaron: {self.people_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Total a bordo: {self.people_in - self.people_out}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Modo: {self.detection_mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar frame
            cv2.imshow("Bus Passenger Counter", frame)
            
            # Guardar frame (opcional)
            # out.write(frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Liberar recursos
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
        
        return self.people_in, self.people_out

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al video o índice de la cámara (0 para webcam)
    video_source = "VID 03-2025 MAQ-03 (P).mp4"  # Cambia por la ruta de tu video
    
    # Crear y ejecutar el contador
    # Opciones para detection_mode: "full_body", "upper_body", "both"
    counter = PersonCounter(video_source, line_position=0.6, detection_mode="upper_body")
    entered, exited = counter.process_video()
    
    print(f"Reporte final:")
    print(f"Pasajeros que subieron: {entered}")
    print(f"Pasajeros que bajaron: {exited}")
    print(f"Total de pasajeros actual: {entered - exited}")