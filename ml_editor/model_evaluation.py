from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import numpy as np
import itertools


def get_confusion_matrix_plot(
    predicted_y,
    true_y,
    classes=None,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.get_cmap("binary"),
    figsize=(10, 10),
):
    """오차 행렬을 반환한다.

    Args:
        predicted_y (_type_): 모델의 예측값
        true_y (_type_): 진짜 레이블 값
        classes (_type_, optional): 양쪽 클래스 이름. Defaults to None.
        normalize (bool, optional): 정규화할지 여부. Defaults to False.
        title (str, optional): 그래프 제목. Defaults to "Confusion matrix".
        cmap (_type_, optional): 사용할 컬러맵. Defaults to plt.get_cmap("binary").
        figsize (tuple, optional): 그림 크기. Defaults to (10, 10).
    """
    if classes is None:
        classes = ["Low quality", "High quality"]

    cm = confusion_matrix(true_y, predicted_y)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    title_obj = plt.title(title, fontsize=30)
    title_obj.set_position([0.5, 1.15])

    plt.colorbar(im)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = ".2f" if normalize else "d"
    thresh = (cm.max() - cm.min()) / 2.0 + cm.min()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 이미지는 x, y (y=0이 위)
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=40,
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)


def get_roc_plot(predicted_proba_y, true_y, tpr_bar=-1, fpr_bar=-1, figsize=(10, 10)):
    """AUC 그래프를 반환한다.

    Args:
        predicted_proba_y (_type_): 각 샘플엗 대한 모델의 예측 확률
        true_y (_type_): 레이블의 진짜 값
        tpr_bar (int, optional): 진짜 양성의 임곗값. Defaults to -1.
        fpr_bar (int, optional): 거짓 양성의 임곗값. Defaults to -1.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 10).
    """

    fpr, tpr, thresholds = roc_curve(true_y, predicted_proba_y)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr,
        tpr,
        lw=1,
        alpha=1,
        color="black",
        label="ROC curve (AUC = %0.2f)" % roc_auc,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="grey",
        label="Chance",
        alpha=1,
    )

    plt.plot(
        [0.01, 0.01, 1],
        [0.01, 0.99, 0.99],
        linestyle=":",
        lw=2,
        color="green",
        label="Perfect model",
        alpha=1,
    )

    if tpr_bar != -1:
        plt.plot(
            [0, 1],
            [tpr_bar, tpr_bar],
            linestyle="-",
            lw=2,
            color="red",
            label="TPR requirement",
            alpha=1,
        )
        plt.fill_between([0, 1], [tpr_bar, tpr_bar], [1, 1], alpha=0, hatch="\\")

    if fpr_bar != -1:
        plt.plot(
            [fpr_bar, fpr_bar],
            [0, 1],
            linestyle="-",
            lw=2,
            color="red",
            label="FPR requirement",
            alpha=1,
        )
        plt.fill_between([fpr_bar, 1], [1, 1], alpha=0, hatch="\\")

    plt.legend(loc="lower right")

    plt.ylabel("True positive rate", fontsize=20)
    plt.xlabel("False positive rate", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def get_calibration_plot(predicted_proba_y, true_y, figsize=(10, 10)):
    """보정 곡선을 반환한다.

    Args:
        predicted_proba_y (_type_): 각 샘플에 대한 모델 예측 확률
        true_y (_type_): 실제 정답 레이블 값
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 10).
    """

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    clf_score = brier_score_loss(true_y, predicted_proba_y, pos_label=true_y.max())
    print("\tBrier: %1.3f" % clf_score)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_y, predicted_proba_y, n_bins=10
    )

    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="black",
        label="%1.3f Brier score (0 is best, 1 is worst)" % clf_score,
    )

    ax2.hist(
        predicted_proba_y,
        range=(0, 1),
        bins=10,
        histtype="step",
        lw=2,
        color="black",
    )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plot")

    ax2.set_title("Probability distribution")
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


def get_metrics(predicted_y, true_y):
    """이진 분류를 위한 표준 지표를 구한다.

    Args:
        predicted_y (_type_): 모델의 예측값
        true_y (_type_): 정답 레이블

    Returns:
        _type_: _description_
    """
    # 진짜 양성 / (진짜 양성 + 가짜 양성)
    precision = precision_score(true_y, predicted_y, pos_label=None, average="weighted")
    # 진짜 양성 / (진짜 양성 + 가짜 음성)
    recall = recall_score(true_y, predicted_y, pos_label=None, average="weighted")

    # 정밀도와 재현율의 조화 평균
    f1 = f1_score(true_y, predicted_y, pos_label=None, average="weighted")

    # 진짜 양성 + 진짜 음성 / 전체
    accuracy = accuracy_score(true_y, predicted_y)
    return accuracy, precision, recall, f1


def get_feature_importance(clf, feature_names):
    """분류기의 특성 중요도를 구한다.

    Args:
        clf (_type_): sklearn 분류기
        feature_names (_type_): 특성 이름 리스트

    Returns:
        _type_: (특성 이름, 점수) 로 구성된 튜플 리스트
    """
    importances = clf.feature_importances_
    indices_sorted_by_importance = np.argsort(importances)[::-1]
    return list(
        zip(
            feature_names[indices_sorted_by_importance],
            importances[indices_sorted_by_importance],
        )
    )


def get_top_k(df, proba_col, true_label_col, k=5, decision_threshold=0.5):
    """이진 분류 문제를 위해 각 클래스 별로 가장 올바른 k개 샘플, 가장 잘못된 k개 샘플,
    가장 불확실한 k개 샘플을 반환한다.

    Args:
        df (_type_): 예측과 진짜 레이블을 담고 이는 DataFrame
        proba_col (_type_): 예측 확률의 열 이름
        true_label_col (_type_): 정답 레이블의 열 이름
        k (int, optional): 각 케이스 당 샘플의 개수. Defaults to 5.
        decision_threshold (float, optional): 양성으로 분류하는 분류기 결정 경계. Defaults to 0.5.

    Returns:
        _type_: correct_pos, correct_neg, incorrect_pos, incorrect_neg, unsure
    """
    # 올바르게 예측한 데이터와 잘못 예측한 데이터를 나눈다.
    correct = df[(df[proba_col] > decision_threshold) == df[true_label_col]].copy()
    incorrect = df[(df[proba_col] > decision_threshold) != df[true_label_col]].copy()

    # 올바르게 양성을 예측한 데이터 중 확률(proba_col)이 가장 높은(가장 큰 확신을 갖는) k개의 샘플을 추출한다.
    top_correct_positive = correct[correct[true_label_col]].nlargest(k, proba_col)
    # 올바르게 음성을 예측한 데이터 중 확률이 가장 낮은 확률(음성에 대한 가장 큰 확신을 갖는) k개의 샘플을 추출한다.
    top_correct_negative = correct[~correct[true_label_col]].nsmallest(k, proba_col)

    # 양성데이터를 음성이라고 예측한 데이터 중
    # 가장 확률이 낮은(가장 음성이라고 확신한) 데이터를 추출한다.
    top_incorrect_positive = incorrect[incorrect[true_label_col]].nsmallest(k, proba_col)
    # 음성데이터를 양성이라고 예측한 데이터 중
    # 가장 확률이 높은(가장 양성이라고 확신한) 데이터를 추출한다.
    top_incorrect_negative = incorrect[~incorrect[true_label_col]].nlargest(k, proba_col)

    # 결정 경계에 가장 가까운 샘플(가장 모호한)을 추출한다.
    most_uncertain = df.iloc[(df[proba_col] - decision_threshold).abs().argsort()[:k]]

    return (
        top_correct_positive,
        top_correct_negative,
        top_incorrect_positive,
        top_incorrect_negative,
        most_uncertain,
    )
