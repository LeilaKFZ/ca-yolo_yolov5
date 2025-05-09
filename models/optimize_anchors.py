import os
import torch.nn as nn
import torchvision.ops.boxes as bops
from sklearn.cluster import KMeans
import numpy as np
import yaml
import itertools
from PIL import Image


class Optimize_Anchors(nn.Module):
    def __init__(
            self,
            annotation_crowdH_path: str = "./testtt/annotations/",
            config_path: str = "yolov5l_ca.yaml",
            image_dir: str = "C:/Users/THINKPAD/PycharmProjects/ca-yolo_yolov5/models/testtt/images",
    ):
        super().__init__()
        self.annotation_crowdH_path = annotation_crowdH_path
        self.config_path = config_path
        self.image_dir = image_dir

        # Vérifier si le répertoire existe
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Le répertoire d'images n'existe pas: {self.image_dir}")

        # ____ obtenir la taille de l'image dynamiquement - accepter plusieurs extensions d'image courantes
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images = [f for f in os.listdir(self.image_dir)
                  if os.path.splitext(f.lower())[1] in valid_extensions]

        if not images:
            raise ValueError(f"Aucune image trouvée dans le répertoire: {self.image_dir}")

        # Détection dynamique de la taille à partir de la première image
        try:
            first_image = images[0]
            img_path = os.path.join(self.image_dir, first_image)
            print(f"Chargement de l'image: {img_path}")
            img = Image.open(img_path)
            self.image_size = max(img.size)  # on prend la plus grande dimension
            print(f"Taille d'image détectée: {self.image_size}")
        except Exception as e:
            print(f"Avertissement: Impossible de lire l'image, utilisation de la valeur par défaut 640. Erreur: {e}")
            self.image_size = 640

        # ______ Infos a partir de yaml5L, possible car: yolo5 not anchor free
        try:
            with open(self.config_path, 'r') as cp:
                config = yaml.safe_load(cp)

            self.anchors = config['anchors']
            self.nb_layers = len(self.anchors)  # selon le yaml c 3
            self.nb_anchor_per_layer = len(self.anchors[0]) // 2  # la aussi c un 3
            self.k = self.nb_layers * self.nb_anchor_per_layer  # k = 9
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier de configuration: {e}")

    def forward(self, x=None):
        boxes = []

        # ____ Extraire les boites (h,w)
        for file in os.listdir(self.annotation_crowdH_path):
            if file.endswith('txt'):
                with open(os.path.join(self.annotation_crowdH_path, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        _, _, _, w_norm, h_norm = map(float, parts)
                        w = w_norm * self.image_size
                        h = h_norm * self.image_size
                        boxes.append([w, h])

        boxes_array = np.array(boxes)  # toutes boites sus forme (w,h) en array pour k-means

        if len(boxes_array) == 0:
            raise ValueError("No valid bounding boxes found in the annotations.")

        # __________________________
        # ______ Apply K-means ___________________
        # __________________________
        kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=10)
        kmeans.fit(boxes_array)

        ####### centre des clusters #######
        centers_anchors = kmeans.cluster_centers_

        # _____ Assigner chaque boite a l'ancre leplus proche ___
        assignements_nearest_id = []

        # Mesurer la distance en utilisant Stat: Ratio
        def ratio_distance(box, anchor):
            epsilon = 1e-5
            w_box, h_box = box
            w_anchor, h_anchor = anchor
            return max(w_box / w_anchor, w_anchor / w_box) * max(h_box / h_anchor, h_anchor / h_box)

        for box in boxes_array:
            distances = [ratio_distance(box, anchor) for anchor in centers_anchors]
            min_dist = min(distances)
            min_indx = distances.index(min_dist)

            # a voir si bien ou tres stricte comme
            if min_dist < 4.0:
                assignements_nearest_id.append(min_indx)
            else:
                continue

        # __________________________
        # ______ Algo Genetique ___________________
        # __________________________

        '''
            1. Population Initiale: Prendre une centaine de modif
            2. Evolution (boucle sur qlq generation)
                2.1. Selection: Best anchors selon "fitness"
                2.2. Croisement: Melanger deux parent -> creer en fant (moy + petit bruit)
                2.3. Mutation: appliquer "bruit gaussien sur (w et h)"
                2.4. Evaluation: Recalculer le fitness de chaque Enfant
            3. Arret: apres N_gen, prendre l'individu le plus performant
        '''

        # 1..... Population Init ..........
        num_variants = 200
        populations = []
        for _ in range(num_variants):
            individual = centers_anchors * np.random.uniform(0.9, 1.1, centers_anchors.shape)
            populations.append(individual)

        # .... SELECTION DU BEST dans POPULATION: calculer "fitness" -> quel anchor est bon, lequel est mauvais
        def iou_wh(box, anchor):
            min_w = min(box[0], anchor[0])
            min_h = min(box[1], anchor[1])
            inter = min_h * min_w
            union = box[0] * box[1] + anchor[0] * anchor[1] - inter
            return inter / union

        def fitness(centers, boxes_array, noise_level=0.01):
            ious = [max(iou_wh(box, anch) for anch in centers) for box in boxes_array]

            # ajout de bruit de (+-)1% a ious afin d'eviter que l'algo se bloque dans un optimum local
            #  -> ca favorise la diversite des parents meme si leur fitness est tres proche
            mean_iou = np.mean(ious)
            noise = np.random.uniform(-noise_level, noise_level)
            return mean_iou * (1 + noise)

        generations = 10
        for gen in range(generations):
            fitness_scores = [fitness(pop, boxes_array, 0.01) for pop in populations]
            fitness_scores = np.array(fitness_scores)
            indices_trie = np.argsort(-fitness_scores)  # ordre décroissant
            num_parents = max(2, int(0.1 * len(populations)))  # top 10 + 2 parents MIN
            top_indices_parents = indices_trie[:num_parents]

            children = []
            selected_parents = [populations[i] for i in top_indices_parents]
            for parent1, parent2 in itertools.combinations(selected_parents, 2):
                child = (parent1 + parent2) / 2 + np.random.normal(0, 0.05, parent1.shape)
                children.append(child)

            # ...... Evaluation ...............
            fitness_children_scores = [fitness(child, boxes_array) for child in children]
            fitness_children_scores = np.array(fitness_children_scores)
            sorted_indices_children = np.argsort(-fitness_children_scores)

            top_n_child = max(1, int(0.1 * len(children)))
            top_id_child = sorted_indices_children[:top_n_child]
            best_children = [children[i] for i in top_id_child]

            # Mise a jour pour generation suivante
            populations = best_children + selected_parents

        # Selection finale du meilleur individu
        final_fitness_scores = [fitness(ind, boxes_array) for ind in populations]
        best_final_index = np.argmax(final_fitness_scores)
        best_final_anchors = populations[best_final_index]

        return {
            "best_children": best_final_anchors,
            "initial_anchors_centers": centers_anchors,
            "assignement_nearest_id": assignements_nearest_id,
        }


# ________________________________________________________
# _______________test _____________________________________
'''
if __name__ == "__main__":
    print("Démarrage de l'optimisation des ancres...")


    optimizer = Optimize_Anchors(
        annotation_crowdH_path = "./testtt/annotations/",
        config_path = "yolov5l_ca.yaml",
        image_dir = "C:/Users/THINKPAD/PycharmProjects/ca-yolo_yolov5/models/testtt/images",
    )

    print("Lancement de l'optimisation...")
    result = optimizer()

    # Affichage structuré des résultats
    import pprint

    pp = pprint.PrettyPrinter(indent=2)
    print("===== Résultats du script =====")
    pp.pprint(result)
'''