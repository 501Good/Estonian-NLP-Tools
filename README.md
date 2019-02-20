# Estonian-NLP-Tools

## Tokenizer

The tokenizer uses a bidirectional LSTM model trained on the [Estonian UD corpus v2.3](https://github.com/UniversalDependencies/UD_Estonian-EDT). 

The corpus was cut into the chunks of five sentences. Each chunk is then was transformed into a sequence of tags for each character: X - out of token, E - end of token, T - end of sentence. 
End of token and end of sentence tags are placed on the last character of a word/sentence. 

The model was trained until it showed no improvement in validation loss for five consecutive epochs (120 epochs in total).

The F1-score reported by the official UD evaluation script is shown in the table below:

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Words</th>
            <th>Sents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>UDPipe</td>
            <td>99.95%</td>
            <td>91.72%	</td>
        </tr>
        <tr>
            <td>LSTM</td>
            <td>99.94%</td>
            <td>90.22%</td>
        </tr>
        <tr>
            <td>ESTNLTK</td>
            <td>99.12%</td>
            <td>85.56%</td>
        </tr>
        <tr>
            <td>Stanford NLP</td>
            <td>99.91%</td>
            <td>93.28%</td>
        </tr>
    </tbody>
</table>


Scores on the test set from the Ettenten corpus: 

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Words</th>
            <th>Sents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>UDPipe</td>
            <td>96.02%</td>
            <td>61.38%</td>
        </tr>
        <tr>
            <td>LSTM</td>
            <td>95.89%</td>
            <td>88.59%</td>
        </tr>
        <tr>
            <td>ESTNLTK</td>
            <td>97.41%</td>
            <td>54.00%</td>
        </tr>
        <tr>
            <td>Stanford NLP</td>
            <td>96.37%</td>
            <td>48.56%</td>
        </tr>
    </tbody>
</table>

## Training on Ettenten corpus

For the training set, I randomly picked one document from each file, because otherwise the training set becomes huge. I trained **three** models. Their name encoded in format `TRAIN SET + DEV SET` where `E` signifies data from the Ettenten corpus and `U` - from the UD corpus. For example, `E+E` means that the train set was taken from the Ettenten data and the dev set was also taken from the Ettenten data.

Ettenten test:

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Words</th>
            <th>Sents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>E+E</td>
            <td>99.33%</td>
            <td>58.86%</td>
        </tr>
        <tr>
            <td>E+U</td>
            <td>99.35%</td>
            <td>58.71%</td>
        </tr>
        <tr>
            <td>U+E</td>
            <td>95.88%</td>
            <td>59.23%</td>
        </tr>
    </tbody>
</table>


UD test:

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Words</th>
            <th>Sents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>E+E</td>
            <td>99.87%</td>
            <td>85.21%</td>
        </tr>
        <tr>
            <td>E+U</td>
            <td>99.85%</td>
            <td>84.36%</td>
        </tr>
        <tr>
            <td>U+E</td>
            <td>99.94%</td>
            <td>89.29%</td>
        </tr>
    </tbody>
</table>
