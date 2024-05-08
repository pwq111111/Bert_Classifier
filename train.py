import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import os

import argparse


# 数据预处理
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = str(self.data.questioncontent[index])
        question = " ".join(question.split())

        inputs = self.tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(self.data.output_index[index], dtype=torch.long),
        }


# 模型搭建
def build_model(num_classes):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=num_classes, output_attentions=False, output_hidden_states=False
    )
    return model


# 模型训练
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


# 模型评估
def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred


def main(args):
    # 读取数据
    data = pd.read_csv(args.input_csv)

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 参数设置
    MAX_LEN = 128
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = len(data["output_index"].unique())
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 准备数据
    train_dataset = CustomDataset(train_data, tokenizer, MAX_LEN)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    test_dataset = CustomDataset(test_data, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES).to(device)

    # 初始化优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练和评估
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = evaluate(model, test_loader, device)
        report = classification_report(y_true, y_pred, output_dict=True)

        # 输出评估结果
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("Train Loss:", train_loss)
        print("Test Precision:", report["macro avg"]["precision"])
        print("Test Recall:", report["macro avg"]["recall"])
        print("Test F1 Score:", report["macro avg"]["f1-score"])

        # 保存模型和评估结果
        save_dir = f"epoch_{epoch + 1}_loss_{train_loss:.4f}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        with open(os.path.join(save_dir, "evaluation.txt"), "w") as f:
            f.write(f"Train Loss: {train_loss}\n")
            f.write(f"Test Precision: {report['macro avg']['precision']}\n")
            f.write(f"Test Recall: {report['macro avg']['recall']}\n")
            f.write(f"Test F1 Score: {report['macro avg']['f1-score']}\n")

        # 保存评估结果到 CSV 文件
        result_df = pd.DataFrame(report).transpose()
        result_df.to_csv(os.path.join(save_dir, "evaluation.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Text Classification")
    parser.add_argument("--input-csv", type=str, default="your_data.csv", help="Input CSV file path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for optimizer")
    args = parser.parse_args()

    main(args)
