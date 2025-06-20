// --- Константы по умолчанию ---
const DEFAULT_INPUT_HEIGHT = 5;
const DEFAULT_INPUT_WIDTH = 5;
const DEFAULT_INPUT_CHANNELS = 3;
const DEFAULT_KERNEL_SIZE = 3;
const DEFAULT_STRIDE = 1;
const DEFAULT_OUTPUT_CHANNELS = 3; // количество выходных каналов

// --- Глобальное состояние ---
window._state = {
    input: {
        height: DEFAULT_INPUT_HEIGHT,
        width: DEFAULT_INPUT_WIDTH,
        channels: DEFAULT_INPUT_CHANNELS,
        data: null,
        selectedChannel: 0,
    },
    conv: {
        channels: DEFAULT_OUTPUT_CHANNELS, // выходные каналы
        kernel: DEFAULT_KERNEL_SIZE, 
        stride: DEFAULT_STRIDE, 
        data: null, 
        selectedChannel: 0,
        kernelWeights: null,
        selectedKernelOutChannel: 0,
        selectedKernelInChannel: 0,
        selectedKernelViewChannel: 0 // для переключения между слоями ядра при просмотре
    },
    output: { data: null, selectedChannel: 0 },
};

// Генерация случайного целого числа в заданном диапазоне
function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Создание 3D тензора с случайными значениями
function createTensor(height, width, channels) {
    const tensor = [];
    for (let c = 0; c < channels; c++) {
        const layer = [];
        for (let h = 0; h < height; h++) {
            const row = [];
            for (let w = 0; w < width; w++) {
                row.push(getRandomInt(-9, 9));
            }
            layer.push(row);
        }
        tensor.push(layer);
    }
    return tensor;
}

// Создание 4D ядра свёртки с случайными значениями
// Размерность: [outChannels][inChannels][kernelHeight][kernelWidth]
function create4DKernel(outChannels, inChannels, kernelHeight, kernelWidth) {
    const kernel = [];
    for (let oc = 0; oc < outChannels; oc++) {
        const outLayer = [];
        for (let ic = 0; ic < inChannels; ic++) {
            const inLayer = [];
            for (let h = 0; h < kernelHeight; h++) {
                const row = [];
                for (let w = 0; w < kernelWidth; w++) {
                    row.push(getRandomInt(-2, 2)); // меньший диапазон для ядер
                }
                inLayer.push(row);
            }
            outLayer.push(inLayer);
        }
        kernel.push(outLayer);
    }
    return kernel;
}

// Функции для создания padding
function applyPadding(tensor, paddingType, paddingSize) {
    // ИСПРАВЛЕНИЕ: При режиме 'valid' padding никогда не применяется
    if (paddingType === 'valid') {
        return tensor; // Без padding, независимо от paddingSize
    }
    
    // Для остальных режимов проверяем размер padding
    if (paddingSize === 0) {
        return tensor; // Без padding
    }
    
    const channels = tensor.length;
    const height = tensor[0].length;
    const width = tensor[0][0].length;
    const padSize = paddingSize;
    
    const paddedTensor = [];
    
    for (let c = 0; c < channels; c++) {
        const paddedLayer = [];
        const newHeight = height + 2 * padSize;
        const newWidth = width + 2 * padSize;
        
        for (let h = 0; h < newHeight; h++) {
            const row = [];
            for (let w = 0; w < newWidth; w++) {
                let value = 0;
                
                // Вычисляем исходные координаты
                const origH = h - padSize;
                const origW = w - padSize;
                
                if (origH >= 0 && origH < height && origW >= 0 && origW < width) {
                    // Внутри исходного тензора
                    value = tensor[c][origH][origW];
                } else {
                    // В области padding
                    if (paddingType === 'reflect') {
                        // Зеркальное отражение БЕЗ дублирования граничных ячеек
                        let mirrorH = origH;
                        let mirrorW = origW;
                        
                        // Обрабатываем отражение по вертикали
                        if (height > 1) {
                            if (mirrorH < 0) {
                                // Отражение через левую границу: -1 -> 1, -2 -> 2, -3 -> 3
                                mirrorH = -mirrorH;
                                // Циклическое отражение для больших padding
                                const cycle = 2 * (height - 1);
                                if (mirrorH >= height) {
                                    mirrorH = mirrorH % cycle;
                                    if (mirrorH >= height) {
                                        mirrorH = cycle - mirrorH;
                                    }
                                }
                            } else if (mirrorH >= height) {
                                // Отражение через правую границу: h -> h-2, h+1 -> h-3, h+2 -> h-4
                                mirrorH = 2 * (height - 1) - mirrorH;
                                // Циклическое отражение для больших padding
                                const cycle = 2 * (height - 1);
                                if (mirrorH < 0) {
                                    mirrorH = Math.abs(mirrorH);
                                    mirrorH = mirrorH % cycle;
                                    if (mirrorH >= height) {
                                        mirrorH = cycle - mirrorH;
                                    }
                                }
                            }
                        } else {
                            mirrorH = 0; // Для массива размера 1
                        }
                        
                        // Обрабатываем отражение по горизонтали
                        if (width > 1) {
                            if (mirrorW < 0) {
                                // Отражение через левую границу: -1 -> 1, -2 -> 2, -3 -> 3
                                mirrorW = -mirrorW;
                                // Циклическое отражение для больших padding
                                const cycle = 2 * (width - 1);
                                if (mirrorW >= width) {
                                    mirrorW = mirrorW % cycle;
                                    if (mirrorW >= width) {
                                        mirrorW = cycle - mirrorW;
                                    }
                                }
                            } else if (mirrorW >= width) {
                                // Отражение через правую границу: w -> w-2, w+1 -> w-3, w+2 -> w-4
                                mirrorW = 2 * (width - 1) - mirrorW;
                                // Циклическое отражение для больших padding
                                const cycle = 2 * (width - 1);
                                if (mirrorW < 0) {
                                    mirrorW = Math.abs(mirrorW);
                                    mirrorW = mirrorW % cycle;
                                    if (mirrorW >= width) {
                                        mirrorW = cycle - mirrorW;
                                    }
                                }
                            }
                        } else {
                            mirrorW = 0; // Для массива размера 1
                        }
                        
                        // Убеждаемся что координаты в границах
                        mirrorH = Math.max(0, Math.min(height - 1, mirrorH));
                        mirrorW = Math.max(0, Math.min(width - 1, mirrorW));
                        
                        value = tensor[c][mirrorH][mirrorW];
                    } else if (paddingType === 'replicate') {
                        // Расширение границы
                        const clampH = Math.max(0, Math.min(height - 1, origH));
                        const clampW = Math.max(0, Math.min(width - 1, origW));
                        value = tensor[c][clampH][clampW];
                    } else if (paddingType === 'circular') {
                        // Циклический сдвиг
                        let wrapH = ((origH % height) + height) % height;
                        let wrapW = ((origW % width) + width) % width;
                        value = tensor[c][wrapH][wrapW];
                    }
                }
                
                row.push(value);
            }
            paddedLayer.push(row);
        }
        paddedTensor.push(paddedLayer);
    }
    
    return paddedTensor;
}

// Вычисление размеров выходного тензора
function calculateOutputSize(inputSize, kernelSize, stride, paddingType, paddingSize) {
    // ИСПРАВЛЕНИЕ: При режиме 'valid' padding игнорируется
    if (paddingType === 'valid') {
        return Math.floor((inputSize - kernelSize) / stride) + 1;
    } else if (paddingSize === 0) {
        // Для других режимов, но без padding
        return Math.floor((inputSize - kernelSize) / stride) + 1;
    } else {
        // Для всех типов padding с размером > 0
        const paddedSize = inputSize + 2 * paddingSize;
        return Math.floor((paddedSize - kernelSize) / stride) + 1;
    }
}

// Выполнение свёртки с 4D ядром
function convolve4D(input, kernel4D, stride, paddingType, paddingSize, outputChannelIdx) {
    // Применяем padding к входному тензору
    const paddedInput = applyPadding(input, paddingType, paddingSize);
    
    const inputHeight = paddedInput[0].length;
    const inputWidth = paddedInput[0][0].length;
    const inputChannels = paddedInput.length;
    const kernelHeight = kernel4D[0][0].length;
    const kernelWidth = kernel4D[0][0][0].length;

    const outputHeight = calculateOutputSize(input[0].length, kernelHeight, stride, paddingType, paddingSize);
    const outputWidth = calculateOutputSize(input[0][0].length, kernelWidth, stride, paddingType, paddingSize);

    const output = Array(outputHeight).fill().map(() => 
        Array(outputWidth).fill(0)
    );

    for (let h = 0; h < outputHeight; h++) {
        for (let w = 0; w < outputWidth; w++) {
            let sum = 0;
            for (let ic = 0; ic < inputChannels; ic++) {
                for (let kh = 0; kh < kernelHeight; kh++) {
                    for (let kw = 0; kw < kernelWidth; kw++) {
                        const inputH = h * stride + kh;
                        const inputW = w * stride + kw;
                        if (inputH < inputHeight && inputW < inputWidth) {
                            sum += paddedInput[ic][inputH][inputW] * kernel4D[outputChannelIdx][ic][kh][kw];
                        }
                    }
                }
            }
            output[h][w] = sum;
        }
    }

    return output;
}



// --- Глобальные переменные для выделения ---
window._highlight = {
    layer: null,
    channel: null,
    i: null,
    j: null,
    fixed: false  // флаг фиксации выделения
};
window._inputShowChannel = null;
window._kernelShowChannel = {}; // для каждого слоя: {layerIdx: channelIdx}
window._clickInProgress = false; // флаг для предотвращения конфликтов между click и mouseleave
window._needsFullRender = false; // флаг для случаев когда нужен полный перерендер

// --- Кэш для быстрого поиска связей ---
window._receptiveCache = new Map();

function getReceptiveFieldCoords(layerIdx, channelIdx, i, j) {
    // Ключ для кэша
    const key = `${layerIdx}|${channelIdx}|${i}|${j}`;
    if (window._receptiveCache.has(key)) {
        return window._receptiveCache.get(key);
    }
    if (layerIdx === 0) {
        const res = [{layer: 0, channel: channelIdx, i, j}];
        window._receptiveCache.set(key, res);
        return res;
    }
    const prev = layerIdx - 1;
    let stride = 1, kernel = 3, paddingType = 'valid', paddingSize = 0;
    if (layerIdx === 1) { 
        stride = window._state.conv.stride; 
        kernel = window._state.conv.kernel;
        paddingType = window._state.conv.padding || 'valid';
        paddingSize = window._state.conv.paddingSize || 0;
    }
    const prevChannels = prev === 0 ? window._state.input.channels : window._state.conv.channels;
    const coords = [];
    
    // Учитываем padding при вычислении координат
    const padSize = (paddingType !== 'valid') ? paddingSize : 0;
    
    for (let c = 0; c < prevChannels; c++) {
        for (let ki = 0; ki < kernel; ki++) {
            for (let kj = 0; kj < kernel; kj++) {
                // С учётом padding координаты сдвигаются
                const paddedI = i * stride + ki;
                const paddedJ = j * stride + kj;
                
                // Конвертируем обратно в координаты исходного тензора (без padding)
                const pi = paddedI - padSize;
                const pj = paddedJ - padSize;
                
                let h = 0, w = 0;
                if (prev === 0) {
                    h = window._state.input.height;
                    w = window._state.input.width;
                } else {
                    h = window._state.conv.data[c].length;
                    w = window._state.conv.data[c][0].length;
                }
                
                // Проверяем что координаты в пределах исходного тензора
                if (pi >= 0 && pi < h && pj >= 0 && pj < w) {
                    coords.push({layer: prev, channel: c, i: pi, j: pj});
                }
            }
        }
    }
    window._receptiveCache.set(key, coords);
    return coords;
}

function isHighlighted(layerIdx, channelIdx, i, j) {
    const h = window._highlight;
    if (h.layer === null) return false;
    if (layerIdx === h.layer && channelIdx === h.channel && i === h.i && j === h.j) return true;
    // Оптимизация: BFS с посещёнными
    const visited = new Set();
    let queue = [{layer: h.layer, channel: h.channel, i: h.i, j: h.j}];
    while (queue.length > 0) {
        const next = [];
        for (const pos of queue) {
            const vkey = `${pos.layer}|${pos.channel}|${pos.i}|${pos.j}`;
            if (visited.has(vkey)) continue;
            visited.add(vkey);
            if (pos.layer === layerIdx && pos.channel === channelIdx && pos.i === i && pos.j === j) return true;
            if (pos.layer > layerIdx) {
                next.push(...getReceptiveFieldCoords(pos.layer, pos.channel, pos.i, pos.j));
            }
        }
        queue = next;
    }
    return false;
}





// --- Глобальный обработчик кликов с делегированием ---
document.addEventListener('click', (event) => {
    
    // Проверяем, кликнули ли по ячейке Conv2d
    if (event.target.classList.contains('cell') && event.target.dataset.layer) {
        const layerIdx = parseInt(event.target.dataset.layer);
        const c = parseInt(event.target.dataset.channel);
        const i = parseInt(event.target.dataset.i);
        const j = parseInt(event.target.dataset.j);
        
        window._clickInProgress = true;
        
        // Обрабатываем клики по conv слою (layer = 1 для единственного conv слоя)
        if (layerIdx === 1) {
            if (window._highlight.layer === layerIdx && 
                window._highlight.channel === c && 
                window._highlight.i === i && 
                window._highlight.j === j) {
                // Переключаем фиксацию
                window._highlight.fixed = !window._highlight.fixed;
                if (!window._highlight.fixed) {
                    window._kernelShowChannel = {};
                    // Сбрасываем выбранный канал ядра
                    window._state.conv.selectedKernelViewChannel = 0;
                    // Полностью сбрасываем выделение при снятии фиксации
                    window._highlight = {layer: null, channel: null, i: null, j: null, fixed: false};
                }
                            } else {
                    // Устанавливаем новое выделение и фиксируем его
                    window._highlight = {layer: layerIdx, channel: c, i, j, fixed: true};
                    window._kernelShowChannel = {};
                    // Синхронизируем каналы при фиксации
                    window._state.input.selectedChannel = 0;
                    window._state.conv.selectedKernelViewChannel = 0;
                }
            
            // Задержка перед перерендером
            setTimeout(() => {
                renderAll();
                window._clickInProgress = false;
            }, 10);
        } else {
            setTimeout(() => {
                window._clickInProgress = false;
            }, 150);
        }
        
        event.stopPropagation();
        return;
    }
    
    // Если клик не по ячейке и выделение не зафиксировано, сбрасываем его
    if (!window._highlight.fixed) {
        if (window._highlight.layer !== null) {
            window._highlight = {layer: null, channel: null, i: null, j: null, fixed: false};
            window._kernelShowChannel = {};
            renderAll();
        }
    }
});

// --- Функция для обновления только стилей выделения без полного перерендера ---
function updateHighlightStyles() {
    // Для сложных изменений (смена каналов, режимов) всё же нужен полный перерендер
    if (window._needsFullRender) {
        window._needsFullRender = false;
        renderAll();
        return;
    }
    
    // Если есть активное выделение, перерендериваем всё для правильной подсветки
    if (window._highlight.layer !== null) {
        renderAll();
        return;
    }
    
    // Обновляем стили всех ячеек без пересоздания элементов
    const cells = document.querySelectorAll('.cell[data-layer]');
    cells.forEach(cell => {
        const layerIdx = parseInt(cell.dataset.layer);
        const c = parseInt(cell.dataset.channel);
        const i = parseInt(cell.dataset.i);
        const j = parseInt(cell.dataset.j);
        
        // Определяем цвет фона
        let backgroundColor = '#fff';
        if (isHighlighted(layerIdx, c, i, j)) {
            backgroundColor = '#ffe066'; // жёлтый для подсвеченных
        }
        if (window._highlight.fixed && 
            window._highlight.layer === layerIdx && 
            window._highlight.channel === c && 
            window._highlight.i === i && 
            window._highlight.j === j) {
            backgroundColor = '#ff5722'; // красно-оранжевый для зафиксированных
            cell.style.color = '#fff';
            cell.style.fontWeight = 'bold';
        } else {
            cell.style.color = '#000';
            cell.style.fontWeight = 'normal';
        }
        cell.style.background = backgroundColor;
    });
    
    // Обновляем входные данные - показываем числа при наведении или выделении
    updateInputDisplay();
    
    // Обновляем conv слои - показываем числа выбранных каналов или при выделении
    updateConvDisplay();
    

}

// --- Функция для обновления отображения входных данных ---
function updateInputDisplay() {
    const inputLayers = document.querySelectorAll('.input-cube-layer');
    const {height, width, channels, data} = window._state.input;
    
    // Проверяем, есть ли активное выделение
    let highlightSet = new Set();
    let highlightMode = false;
    if (window._highlight && window._highlight.layer !== null && window._highlight.layer > 0) {
        const allInfluencing = getAllInfluencingCells(
            window._highlight.layer, 
            window._highlight.channel, 
            window._highlight.i, 
            window._highlight.j
        );
        if (allInfluencing.has(0)) {
            highlightSet = allInfluencing.get(0);
        }
        highlightMode = true;
    }
    
    inputLayers.forEach((layerDiv) => {
        const c = parseInt(layerDiv.dataset.channel); // Получаем реальный номер канала
        // Очищаем существующие ячейки
        const existingCells = layerDiv.querySelectorAll('.cell');
        existingCells.forEach(cell => cell.remove());
        
        if (highlightMode) {
            // Показываем только влияющие числа
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const key = `${c}|${i}|${j}`;
                    if (highlightSet.has(key)) {
                        const cell = document.createElement('div');
                        cell.className = 'cell';
                        cell.textContent = data[c][i][j];
                        cell.style.position = 'absolute';
                        cell.style.left = `${j * 32}px`;
                        cell.style.top = `${i * 32}px`;
                        cell.style.width = '32px';
                        cell.style.height = '32px';
                        cell.style.display = 'flex';
                        cell.style.alignItems = 'center';
                        cell.style.justifyContent = 'center';
                        cell.style.fontSize = '15px';
                        cell.style.background = '#ffe066';
                        cell.style.border = '1px solid #ccc';
                        cell.style.cursor = 'pointer';
                        cell.style.pointerEvents = 'auto';
                        
                        // Добавляем data-атрибуты (для входных данных layer = 0)
                        cell.dataset.layer = 0;
                        cell.dataset.channel = c;
                        cell.dataset.i = i;
                        cell.dataset.j = j;
                        
                        cell.onclick = () => {
                            const val = prompt('Новое значение:', data[c][i][j]);
                            if (val !== null) {
                                data[c][i][j] = parseInt(val) || 0;
                                updateAll();
                            }
                        };
                        layerDiv.appendChild(cell);
                    }
                }
            }
        } else if (window._inputShowChannel === c) {
            // Показываем числа наведённого слоя
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.textContent = data[c][i][j];
                    cell.style.position = 'absolute';
                    cell.style.left = `${j * 32}px`;
                    cell.style.top = `${i * 32}px`;
                    cell.style.width = '32px';
                    cell.style.height = '32px';
                    cell.style.display = 'flex';
                    cell.style.alignItems = 'center';
                    cell.style.justifyContent = 'center';
                    cell.style.fontSize = '15px';
                    cell.style.background = '#fff';
                    cell.style.border = '1px solid #ccc';
                    cell.style.cursor = 'pointer';
                    cell.style.pointerEvents = 'auto';
                    
                    // Добавляем data-атрибуты (для входных данных layer = 0)
                    cell.dataset.layer = 0;
                    cell.dataset.channel = c;
                    cell.dataset.i = i;
                    cell.dataset.j = j;
                    
                    cell.onclick = () => {
                        const val = prompt('Новое значение:', data[c][i][j]);
                        if (val !== null) {
                            data[c][i][j] = parseInt(val) || 0;
                            updateAll();
                        }
                    };
                    layerDiv.appendChild(cell);
                }
            }
        }
    });
}

// --- Функция для обновления отображения conv слоёв ---
function updateConvDisplay() {
    // Обновляем единственный conv слой (layerIdx = 0 для conv)
    const convLayers = document.querySelectorAll('.conv-cube-layer[data-layer="0"]');
    const conv = window._state.conv;
    const data = conv.data;
    const channels = conv.channels;
    const height = data[0].length;
    const width = data[0][0].length;
    
    // Проверяем режим выделения
    let highlightSet = new Set();
    let highlightMode = false;
    if (window._highlight && window._highlight.layer !== null && window._highlight.layer > 1) {
        const allInfluencing = getAllInfluencingCells(
            window._highlight.layer, 
            window._highlight.channel, 
            window._highlight.i, 
            window._highlight.j
        );
        if (allInfluencing.has(1)) {
            highlightSet = allInfluencing.get(1);
        }
        highlightMode = true;
    }
    
    convLayers.forEach((layerDiv) => {
        const c = parseInt(layerDiv.dataset.channel); // Получаем реальный номер канала
        // Очищаем существующие ячейки
        const existingCells = layerDiv.querySelectorAll('.cell');
        existingCells.forEach(cell => cell.remove());
        
        if (highlightMode) {
            // Показываем только влияющие числа
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const key = `${c}|${i}|${j}`;
                    if (highlightSet.has(key)) {
                        const cell = document.createElement('div');
                        cell.className = 'cell';
                        cell.textContent = data[c][i][j];
                        cell.style.position = 'absolute';
                        cell.style.left = `${j * 32}px`;
                        cell.style.top = `${i * 32}px`;
                        cell.style.width = '32px';
                        cell.style.height = '32px';
                        cell.style.display = 'flex';
                        cell.style.alignItems = 'center';
                        cell.style.justifyContent = 'center';
                        cell.style.fontSize = '15px';
                        cell.style.background = '#ffe066';
                        cell.style.border = '1px solid #ccc';
                        
                        // Добавляем data-атрибуты для кликабельности
                        cell.dataset.layer = 1;
                        cell.dataset.channel = c;
                        cell.dataset.i = i;
                        cell.dataset.j = j;
                        cell.style.cursor = 'pointer';
                        cell.style.pointerEvents = 'auto';
                        
                        layerDiv.appendChild(cell);
                    }
                }
            }
        } else if (c === conv.selectedChannel || window._kernelShowChannel[0] === c) {
            // Показываем числа выбранного канала или канала при наведении на ядро
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.textContent = data[c][i][j];
                    cell.style.position = 'absolute';
                    cell.style.left = `${j * 32}px`;
                    cell.style.top = `${i * 32}px`;
                    cell.style.width = '32px';
                    cell.style.height = '32px';
                    cell.style.display = 'flex';
                    cell.style.alignItems = 'center';
                    cell.style.justifyContent = 'center';
                    cell.style.fontSize = '15px';
                    
                    // Добавляем data-атрибуты для делегирования событий
                    cell.dataset.layer = 1;
                    cell.dataset.channel = c;
                    cell.dataset.i = i;
                    cell.dataset.j = j;

                    // Определяем цвет фона
                    let backgroundColor = '#fff';
                    if (isHighlighted(1, c, i, j)) {
                        backgroundColor = '#ffe066';
                    }
                    if (window._highlight.fixed && 
                        window._highlight.layer === 1 && 
                        window._highlight.channel === c && 
                        window._highlight.i === i && 
                        window._highlight.j === j) {
                        backgroundColor = '#ff5722';
                        cell.style.color = '#fff';
                        cell.style.fontWeight = 'bold';
                    }
                    cell.style.background = backgroundColor;
                    cell.style.border = '2px solid #666';
                    cell.style.cursor = 'pointer';
                    cell.style.pointerEvents = 'auto';
                    cell.style.zIndex = 1000;
                    cell.style.transition = 'all 0.2s ease';
                    cell.style.userSelect = 'none';
                    
                                            // Обработчики событий только для видимых ячеек
                        if (c === conv.selectedChannel) {
                            // Добавляем hover-эффект
                            cell.addEventListener('mouseenter', () => {
                                if (!window._highlight.fixed || 
                                    !(window._highlight.layer === 1 && 
                                      window._highlight.channel === c && 
                                      window._highlight.i === i && 
                                      window._highlight.j === j)) {
                                    cell.style.transform = 'scale(1.05)';
                                    cell.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
                                }
                            });
                            
                            cell.addEventListener('mouseleave', () => {
                                if (!window._highlight.fixed || 
                                    !(window._highlight.layer === 1 && 
                                      window._highlight.channel === c && 
                                      window._highlight.i === i && 
                                      window._highlight.j === j)) {
                                    cell.style.transform = 'scale(1)';
                                    cell.style.boxShadow = 'none';
                                }
                            });
                            
                            // Обработчик кликов
                            cell.addEventListener('click', (event) => {
                                event.stopPropagation();
                                window._clickInProgress = true;
                                
                                if (window._highlight.layer === 1 && 
                                    window._highlight.channel === c && 
                                    window._highlight.i === i && 
                                    window._highlight.j === j) {
                                    // Переключаем фиксацию
                                    window._highlight.fixed = !window._highlight.fixed;
                                    if (!window._highlight.fixed) {
                                        window._kernelShowChannel = {};
                                        window._state.conv.selectedKernelViewChannel = 0;
                                        // Полностью сбрасываем выделение при снятии фиксации
                                        window._highlight = {layer: null, channel: null, i: null, j: null, fixed: false};
                                    }
                                } else {
                                    // Устанавливаем новое выделение и фиксируем его
                                    window._highlight = {layer: 1, channel: c, i, j, fixed: true};
                                    window._kernelShowChannel = {};
                                    // Синхронизируем каналы при фиксации
                                    window._state.input.selectedChannel = 0;
                                    window._state.conv.selectedKernelViewChannel = 0;
                                }
                                
                                // Задержка перед перерендером
                                setTimeout(() => {
                                    renderAll();
                                    window._clickInProgress = false;
                                }, 10);
                            });
                        }
                    
                    layerDiv.appendChild(cell);
                }
            }
        }
    });
}



// --- Функция для сброса всех выделений ---
function clearAllHighlights() {
    window._highlight = {layer: null, channel: null, i: null, j: null, fixed: false};
    window._kernelShowChannel = {};
    window._inputShowChannel = null;
    window._clickInProgress = false;
    renderAll();
}

// --- Рендер переключателя каналов ---
function renderChannelSelector(container, channels, selected, onChange) {
    container.innerHTML = '';
    const prev = document.createElement('button');
    prev.textContent = '⟨';
    prev.disabled = selected === 0;
    prev.onclick = () => onChange(selected - 1);
    const next = document.createElement('button');
    next.textContent = '⟩';
    next.disabled = selected === channels - 1;
    next.onclick = () => onChange(selected + 1);
    const label = document.createElement('span');
    label.textContent = `Канал ${selected + 1} / ${channels}`;
    label.style.margin = '0 8px';
    container.appendChild(prev);
    container.appendChild(label);
    container.appendChild(next);
}

// --- 3D визуализация входного тензора с отображением padding ---
function renderInputTensor3D(container) {
    container.innerHTML = '';
    const {height, width, channels, data} = window._state.input;
    
    // Получаем параметры padding
    const paddingType = window._state.conv.padding || 'valid';
    const paddingSize = window._state.conv.paddingSize || 0;
    // ИСПРАВЛЕНИЕ: Показываем padding только если режим НЕ 'valid' И размер > 0
    const showPadding = paddingType !== 'valid' && paddingSize > 0;
    
    // Создаём padded тензор для отображения
    let displayTensor = data;
    let displayHeight = height;
    let displayWidth = width;
    
    if (showPadding) {
        displayTensor = applyPadding(data, paddingType, paddingSize);
        displayHeight = height + 2 * paddingSize;
        displayWidth = width + 2 * paddingSize;
    }
    
    // Проверяем, есть ли активное выделение числа в Conv2d
    let highlightSet = new Set();
    let highlightMode = false;
    if (window._highlight && window._highlight.layer !== null && window._highlight.layer > 0) {
        // Получаем все влияющие ячейки через всю цепочку
        const allInfluencing = getAllInfluencingCells(
            window._highlight.layer, 
            window._highlight.channel, 
            window._highlight.i, 
            window._highlight.j
        );
        
        // Для входного слоя (layer 0)
        if (allInfluencing.has(0)) {
            highlightSet = allInfluencing.get(0);
        }
        highlightMode = true;
    }

    // Фиксированная область для селектора каналов
    const selectorArea = document.createElement('div');
    selectorArea.style.height = '50px'; // Увеличенная фиксированная высота для предотвращения сдвигов
    selectorArea.style.marginBottom = '10px';
    selectorArea.style.display = 'flex';
    selectorArea.style.justifyContent = 'center';
    selectorArea.style.alignItems = 'center';
    selectorArea.style.width = '100%';
    selectorArea.style.flexWrap = 'wrap';
    selectorArea.style.gap = '4px';
    selectorArea.style.minHeight = '50px'; // Минимальная высота для стабильности
    
    if (window._highlight && window._highlight.fixed && window._highlight.layer === 1) {
        // Компактная информация о фиксации
        const fixedBadge = document.createElement('span');
        fixedBadge.style.background = '#ff5722';
        fixedBadge.style.color = '#fff';
        fixedBadge.style.padding = '3px 6px';
        fixedBadge.style.borderRadius = '10px';
        fixedBadge.style.fontSize = '11px';
        fixedBadge.style.fontWeight = 'bold';
        fixedBadge.style.whiteSpace = 'nowrap';
        fixedBadge.textContent = `🔒 К${window._highlight.channel + 1}`;
        selectorArea.appendChild(fixedBadge);
        
        const channelLabel = document.createElement('span');
        channelLabel.textContent = 'Слой:';
        channelLabel.style.color = '#1976d2';
        channelLabel.style.fontWeight = 'bold';
        channelLabel.style.fontSize = '12px';
        channelLabel.style.whiteSpace = 'nowrap';
        selectorArea.appendChild(channelLabel);
        
        renderChannelSelector(selectorArea, channels, window._state.input.selectedChannel, (idx) => {
            // Синхронизируем оба селектора
            window._state.input.selectedChannel = idx;
            window._state.conv.selectedKernelViewChannel = idx;
            renderAll();
        });
    } else if (showPadding) {
        // Селектор каналов для просмотра padding
        renderChannelSelector(selectorArea, channels, window._state.input.selectedChannel, (idx) => {
            window._state.input.selectedChannel = idx;
            renderAll();
        });
    }
    
    container.appendChild(selectorArea);

    const cubeDiv = document.createElement('div');
    cubeDiv.style.position = 'relative';
    cubeDiv.style.width = `${displayWidth * 32 + channels * 18}px`;
    cubeDiv.style.height = `${displayHeight * 32 + channels * 18}px`;
    cubeDiv.style.margin = '0 auto';
    cubeDiv.style.display = 'block';
    cubeDiv.style.cursor = 'pointer';
    // Фиксированные размеры для предотвращения дёрганья
    cubeDiv.style.minWidth = `${displayWidth * 32 + channels * 18}px`;
    cubeDiv.style.minHeight = `${displayHeight * 32 + channels * 18}px`;

    for (let c = 0; c < channels; c++) {
        const layerDiv = document.createElement('div');
        layerDiv.style.position = 'absolute';
        layerDiv.style.left = `${c * 18}px`;
        layerDiv.style.top = `${c * 18}px`;
        layerDiv.style.opacity = '0.85';
        layerDiv.style.background = `rgba(${100 + c*50},${200-c*60},255,0.18)`;
        layerDiv.style.border = '1.5px solid #bbb';
        layerDiv.style.borderRadius = '6px';
        layerDiv.style.width = `${displayWidth * 32}px`;
        layerDiv.style.height = `${displayHeight * 32}px`;
        layerDiv.style.transition = 'box-shadow 0.2s, background 0.2s';
        layerDiv.className = 'input-cube-layer';
        layerDiv.dataset.channel = c;
        
        if (highlightMode) {
            // Режим куба без чисел: отключаем наведение на слои
            layerDiv.style.pointerEvents = 'none';
            
            // Показываем только влияющие числа для всех каналов или выбранного канала при фиксации
            const shouldShowNumbers = window._highlight.fixed ? 
                (c === window._state.input.selectedChannel) : true;
                
            if (shouldShowNumbers) {
                for (let i = 0; i < displayHeight; i++) {
                    for (let j = 0; j < displayWidth; j++) {
                        // Определяем координаты в исходном тензоре (без padding)
                        const origI = showPadding ? (i - paddingSize) : i;
                        const origJ = showPadding ? (j - paddingSize) : j;
                        
                        // Ключ для проверки в highlightSet всегда должен быть с исходными координатами
                        // но нам нужно проверить и padding ячейки
                        let key = `${c}|${origI}|${origJ}`;
                        let shouldHighlight = false;
                        
                        if (showPadding) {
                            // При наличии padding нужно проверить влияет ли эта ячейка на результат
                            // Если это padding ячейка, то может влиять только если padding используется в свёртке
                            const isPaddingCell = origI < 0 || origI >= height || origJ < 0 || origJ >= width;
                            if (isPaddingCell) {
                                // Для padding ячеек создаём специальный ключ
                                key = `${c}|padding|${i}|${j}`;
                                // Проверяем влияет ли padding ячейка - используем рецептивное поле
                                shouldHighlight = isInputCellInfluencing(c, i, j);
                            } else {
                                // Для обычных ячеек используем стандартную проверку
                                shouldHighlight = highlightSet.has(key);
                            }
                        } else {
                            shouldHighlight = highlightSet.has(key);
                        }
                        
                        if (shouldHighlight) {
                            const cell = document.createElement('div');
                            cell.className = 'cell';
                            cell.textContent = displayTensor[c][i][j];
                            cell.style.position = 'absolute';
                            cell.style.left = `${j * 32}px`;
                            cell.style.top = `${i * 32}px`;
                            cell.style.width = '32px';
                            cell.style.height = '32px';
                            cell.style.display = 'flex';
                            cell.style.alignItems = 'center';
                            cell.style.justifyContent = 'center';
                            cell.style.fontSize = '15px';
                            
                            // Определяем цвет фона в зависимости от типа ячейки
                            let backgroundColor = '#ffe066'; // жёлтый для влияющих
                            let borderColor = '#ccc';
                            
                            // Проверяем, является ли ячейка padding
                            const isPaddingCell = showPadding && (origI < 0 || origI >= height || origJ < 0 || origJ >= width);
                            if (isPaddingCell) {
                                backgroundColor = '#ffccbc'; // светло-оранжевый для padding
                                borderColor = '#ff9800';
                                cell.style.fontStyle = 'italic';
                                cell.title = `Padding ячейка (${paddingType})`;
                            } else {
                                cell.title = `Исходная ячейка [${origI}, ${origJ}]`;
                            }
                            
                            cell.style.background = backgroundColor;
                            cell.style.border = `1px solid ${borderColor}`;
                            cell.style.cursor = 'pointer';
                            
                            // Обработчик клика - только для исходных ячеек
                            if (!isPaddingCell) {
                                cell.onclick = () => {
                                    const val = prompt('Новое значение:', data[c][origI][origJ]);
                                    if (val !== null) {
                                        data[c][origI][origJ] = parseInt(val) || 0;
                                        updateAll();
                                    }
                                };
                            }
                            
                            layerDiv.appendChild(cell);
                        }
                    }
                }
            }
        } else {
            // Нормальный режим: показываем все ячейки включая padding
            layerDiv.style.pointerEvents = 'none';
            layerDiv.style.zIndex = c;
            
            // Показываем все числа если включён padding
            if (showPadding && c === window._state.input.selectedChannel) {
                for (let i = 0; i < displayHeight; i++) {
                    for (let j = 0; j < displayWidth; j++) {
                        const origI = i - paddingSize;
                        const origJ = j - paddingSize;
                        const isPaddingCell = origI < 0 || origI >= height || origJ < 0 || origJ >= width;
                        
                        const cell = document.createElement('div');
                        cell.className = 'cell';
                        cell.textContent = displayTensor[c][i][j];
                        cell.style.position = 'absolute';
                        cell.style.left = `${j * 32}px`;
                        cell.style.top = `${i * 32}px`;
                        cell.style.width = '32px';
                        cell.style.height = '32px';
                        cell.style.display = 'flex';
                        cell.style.alignItems = 'center';
                        cell.style.justifyContent = 'center';
                        cell.style.fontSize = '14px';
                        
                        if (isPaddingCell) {
                            cell.style.background = '#ffccbc'; // светло-оранжевый для padding
                            cell.style.border = '1px solid #ff9800';
                            cell.style.fontStyle = 'italic';
                            cell.style.color = '#d84315';
                            cell.title = `Padding ячейка (${paddingType})`;
                        } else {
                            cell.style.background = '#fff';
                            cell.style.border = '1px solid #ccc';
                            cell.style.cursor = 'pointer';
                            cell.title = `Исходная ячейка [${origI}, ${origJ}]`;
                            cell.onclick = () => {
                                const val = prompt('Новое значение:', data[c][origI][origJ]);
                                if (val !== null) {
                                    data[c][origI][origJ] = parseInt(val) || 0;
                                    updateAll();
                                }
                            };
                        }
                        
                        layerDiv.appendChild(cell);
                    }
                }
            }
        }
        cubeDiv.appendChild(layerDiv);
    }
    container.appendChild(cubeDiv);
    
    // Подписи каналов с информацией о padding
    const chSel = document.createElement('div');
    chSel.style.textAlign = 'center';
    chSel.style.marginTop = '8px';
    chSel.style.height = '40px'; // Фиксированная высота для предотвращения сдвигов
    chSel.style.minHeight = '40px';
    chSel.style.display = 'flex';
    chSel.style.alignItems = 'center';
    chSel.style.justifyContent = 'center';
    
    if (window._highlight && window._highlight.fixed && window._highlight.layer === 1) {
        chSel.innerHTML = `Слой ${window._state.input.selectedChannel + 1}/${channels}`;
        chSel.style.color = '#1976d2';
        chSel.style.fontWeight = 'bold';
        chSel.style.fontSize = '12px';
    } else if (highlightMode) {
        chSel.textContent = 'Показаны только влияющие числа';
        chSel.style.fontSize = '12px';
    } else if (showPadding) {
        chSel.innerHTML = `Размер: ${height}×${width} → ${displayHeight}×${displayWidth} <span style="color: #ff9800;">(${paddingType} padding ${paddingSize})</span>`;
        chSel.style.fontSize = '12px';
    } else {
        chSel.textContent = 'Кликните на выходную ячейку для анализа связей';
        chSel.style.fontSize = '12px';
    }
    
    container.appendChild(chSel);
}



// Инициализация при загрузке страницы
window.onload = () => {
    // Переопределяем старую функцию updateVisualization глобально
    window.updateVisualization = applyParamsFromForm;
    
    // Убеждаемся что контейнер main-visualization существует
    const mainVis = document.getElementById('main-visualization');
    if (!mainVis) {
        const mainDiv = document.createElement('div');
        mainDiv.id = 'main-visualization';
        mainDiv.style.width = '100%';
        mainDiv.style.minHeight = '400px';
        mainDiv.style.overflowX = 'auto';
        document.body.appendChild(mainDiv);
    }
    
    // Устанавливаем правильное состояние поля padding при загрузке
    togglePaddingSize();
    
    // Инициализируем архитектуру с параметрами из формы
    applyParamsFromForm();
};

// --- Свёртка для одного слоя с 4D ядром ---
function convolveLayer(input, kernelSize, stride, outChannels, kernel4D = null, paddingType = 'valid', paddingSize = 0) {
    const inChannels = input.length;
    
    // Создаём 4D ядро если не передано
    let kernels = kernel4D;
    if (!kernels) {
        kernels = create4DKernel(outChannels, inChannels, kernelSize, kernelSize);
    }
    
    // Выходной тензор: [outChannels][outHeight][outWidth] 
    const output = [];
    for (let oc = 0; oc < outChannels; oc++) {
        const outputChannel = convolve4D(input, kernels, stride, paddingType, paddingSize, oc);
        output.push(outputChannel);
    }
    
    return { output, kernels };
}



// --- Основная функция визуализации ---
function renderAll() {
    const root = document.getElementById('main-visualization');
    root.innerHTML = '';
    root.style.display = 'flex';
    root.style.flexDirection = 'row';
    root.style.gap = '16px';
    root.style.width = '100%';
    root.style.overflowX = 'auto';
    root.style.padding = '10px';

    // Увеличиваем минимальную ширину если есть большой padding
    const paddingSize = window._state.conv.paddingSize || 0;
    const minWidth = paddingSize > 1 ? '1400px' : '1200px'; // Увеличиваем для новой области
    root.style.minWidth = minWidth;

    // Контейнер для основной визуализации (входные данные, ядро, выходные данные)
    const mainContainer = document.createElement('div');
    mainContainer.style.display = 'flex';
    mainContainer.style.flexDirection = 'row';
    mainContainer.style.gap = '16px';
    mainContainer.style.flex = '1';

    // Входной тензор (3D) - увеличиваем ширину для больших padding
    const inputDiv = document.createElement('div');
    const inputWidth = paddingSize > 1 ? '320px' : '280px';
    inputDiv.style.flex = `0 0 ${inputWidth}`; // Фиксированная ширина
    inputDiv.style.width = inputWidth;    // Явно задаем ширину
    inputDiv.style.display = 'flex';
    inputDiv.style.flexDirection = 'column';
    inputDiv.style.alignItems = 'center';
    const inputTitle = document.createElement('div');
    inputTitle.textContent = 'Входные данные X';
    inputTitle.style.fontWeight = 'bold';
    inputTitle.style.marginBottom = '6px';
    inputTitle.style.color = '#1976d2';
    inputTitle.style.textAlign = 'center';
    inputTitle.style.width = '100%';
    inputDiv.appendChild(inputTitle);
    const inputCubeDiv = document.createElement('div');
    renderInputTensor3D(inputCubeDiv);
    inputDiv.appendChild(inputCubeDiv);
    mainContainer.appendChild(inputDiv);

    // Единственный Conv2d слой с ядром
    // Ядро свёртки
    const kernelDiv = document.createElement('div');
    kernelDiv.style.flex = '0 0 300px'; // Фиксированная ширина
    kernelDiv.style.width = '300px';    // Явно задаем ширину
    kernelDiv.style.display = 'flex';
    kernelDiv.style.flexDirection = 'column';
    kernelDiv.style.alignItems = 'center';
    kernelDiv.style.background = '#fff3e0';
    kernelDiv.style.border = '2px solid #ff9800';
    kernelDiv.style.borderRadius = '8px';
    kernelDiv.style.padding = '10px';
    const kernelContainer = document.createElement('div');
    renderKernel4D(kernelContainer);
    kernelDiv.appendChild(kernelContainer);
    mainContainer.appendChild(kernelDiv);
    
    // Conv2d слой (результат) - увеличиваем ширину для больших padding
    const convDiv = document.createElement('div');
    const outputWidth = paddingSize > 1 ? '320px' : '280px';
    convDiv.style.flex = `0 0 ${outputWidth}`; // Фиксированная ширина
    convDiv.style.width = outputWidth;    // Явно задаем ширину
    convDiv.style.display = 'flex';
    convDiv.style.flexDirection = 'column';
    convDiv.style.alignItems = 'center';
    const convTitle = document.createElement('div');
    convTitle.textContent = 'Y = Conv2d(X, K)';
    convTitle.style.fontWeight = 'bold';
    convTitle.style.marginBottom = '6px';
    convTitle.style.color = '#d32f2f';
    convTitle.style.textAlign = 'center';
    convTitle.style.width = '100%';
    convDiv.appendChild(convTitle);
    const convCubeDiv = document.createElement('div');
    renderConvTensor3D(convCubeDiv);
    convDiv.appendChild(convCubeDiv);
    mainContainer.appendChild(convDiv);

    // Добавляем основной контейнер в root
    root.appendChild(mainContainer);

    // Отдельная область для отображения скалярного произведения
    const calculationDiv = document.createElement('div');
    calculationDiv.style.flex = '0 0 350px'; // Фиксированная ширина
    calculationDiv.style.width = '350px';
    calculationDiv.style.display = 'flex';
    calculationDiv.style.flexDirection = 'column';
    calculationDiv.style.background = '#f8f9fa';
    calculationDiv.style.border = '2px solid #dee2e6';
    calculationDiv.style.borderRadius = '8px';
    calculationDiv.style.padding = '15px';
    calculationDiv.style.maxHeight = '600px';
    calculationDiv.style.overflowY = 'auto';

    // Заголовок области вычислений
    const calcTitle = document.createElement('div');
    calcTitle.textContent = 'Детальные вычисления';
    calcTitle.style.fontWeight = 'bold';
    calcTitle.style.fontSize = '16px';
    calcTitle.style.color = '#495057';
    calcTitle.style.marginBottom = '10px';
    calcTitle.style.textAlign = 'center';
    calcTitle.style.borderBottom = '2px solid #dee2e6';
    calcTitle.style.paddingBottom = '8px';
    calculationDiv.appendChild(calcTitle);

    // Контейнер для содержимого вычислений
    const calcContent = document.createElement('div');
    calcContent.id = 'calculation-content';
    
    // Проверяем есть ли зафиксированное выделение
    if (window._highlight && window._highlight.fixed && window._highlight.layer === 1) {
        const calculation = generateConvolutionCalculation(
            window._highlight.channel, 
            window._highlight.i, 
            window._highlight.j
        );
        calcContent.innerHTML = calculation;
    } else {
        calcContent.innerHTML = `
            <div style="text-align: center; color: #6c757d; font-style: italic; margin-top: 50px;">
                <div style="font-size: 48px; margin-bottom: 15px; width: 48px; height: 48px; margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center;">🔍</div>
                <div style="font-size: 14px; line-height: 1.5;">
                    Кликните на выходную ячейку<br>
                    для просмотра детальных<br>
                    вычислений скалярного<br>
                    произведения
                </div>
            </div>
        `;
    }
    
    calculationDiv.appendChild(calcContent);
    root.appendChild(calculationDiv);
}

// --- Пересчёт всех слоёв ---
function updateAll() {
    // Вход
    if (!window._state.input.data) {
        window._state.input.data = createTensor(
            window._state.input.height,
            window._state.input.width,
            window._state.input.channels
        );
    }
    
    // Единственный Conv2d слой
    const conv1 = convolveLayer(
        window._state.input.data,
        window._state.conv.kernel,
        window._state.conv.stride,
        window._state.conv.channels,
        window._state.conv.kernelWeights,
        window._state.conv.padding,
        window._state.conv.paddingSize
    );
    window._state.conv.data = conv1.output;
    window._state.conv.kernelWeights = conv1.kernels;
    
    // Выход (результат свёртки)
    window._state.output.data = conv1.output;
    renderAll();
}



// --- Обработка параметров из формы ---
function applyParamsFromForm() {
    
    // Получаем значения из формы
    const inputHeight = parseInt(document.getElementById('inputHeight').value);
    const inputWidth = parseInt(document.getElementById('inputWidth').value);
    const inputChannels = parseInt(document.getElementById('inputChannels').value); // один ползунок для входных каналов
    const kernelSize = parseInt(document.getElementById('kernelHeight').value); // считаем квадратное ядро
    const outputChannels = parseInt(document.getElementById('outputChannels').value); // выходные каналы
    const stride = parseInt(document.getElementById('stride').value);
    const paddingType = document.getElementById('paddingType').value;
    const paddingSize = parseInt(document.getElementById('paddingSize').value);

    // ИСПРАВЛЕНИЕ: При режиме 'valid' размер padding уже установлен в 0 через togglePaddingSize
    let validPaddingSize = paddingSize;
    if (paddingType === 'valid') {
        validPaddingSize = 0;
        
        // Проверяем что ядро не больше входного тензора в режиме Valid
        if (kernelSize > inputHeight || kernelSize > inputWidth) {
            alert(`⚠️ Ошибка: В режиме Valid размер ядра (${kernelSize}×${kernelSize}) не может быть больше размера входного тензора (${inputHeight}×${inputWidth})!`);
            return;
        }
    } else {
        // Валидация размера padding - должен быть меньше размера ядра (только для не-valid режимов)
        const maxPaddingSize = kernelSize - 1;
        
        if (paddingSize >= kernelSize) {
            validPaddingSize = maxPaddingSize;
            alert(`⚠️ Ошибка: Размер padding (${paddingSize}) должен быть меньше размера ядра (${kernelSize}×${kernelSize})!\n\nРазмер padding скорректирован на ${validPaddingSize}.`);
            document.getElementById('paddingSize').value = validPaddingSize;
        }
        
        // Проверяем что ядро не больше расширенного тензора
        const paddedHeight = inputHeight + 2 * validPaddingSize;
        const paddedWidth = inputWidth + 2 * validPaddingSize;
        
        if (kernelSize > paddedHeight || kernelSize > paddedWidth) {
            alert(`⚠️ Ошибка: Размер ядра (${kernelSize}×${kernelSize}) не может быть больше размера расширенного тензора (${paddedHeight}×${paddedWidth})!\n\nУвеличьте размер входного тензора или уменьшите размер ядра.`);
            return;
        }
    }



    // Входные каналы ядра автоматически равны входным каналам тензора
    const kernelDepth = inputChannels;

    // Сброс состояния
    window._state = {
        input: {
            height: inputHeight,
            width: inputWidth,
            channels: inputChannels,
            data: createTensor(inputHeight, inputWidth, inputChannels),
            selectedChannel: 0,
        },
        conv: {
            channels: outputChannels, // используем выходные каналы из формы
            kernel: kernelSize, 
            stride: stride, 
            padding: paddingType,
            paddingSize: validPaddingSize,
            data: null, 
            selectedChannel: 0,
            kernelWeights: null,
            selectedKernelOutChannel: 0,
            selectedKernelInChannel: 0,
            selectedKernelViewChannel: 0 // для переключения между слоями ядра при просмотре
        },
        output: { data: null, selectedChannel: 0 },
    };
    window._receptiveCache = new Map();
    
    // Сбрасываем выделение при смене параметров
    window._highlight = {layer: null, channel: null, i: null, j: null, fixed: false};
    
    updateAll();
}

// --- Переопределяю кнопку "Обновить визуализацию" ---
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.querySelector('button[onclick="updateVisualization()"]');
    if (btn) {
        btn.onclick = (e) => {
            e.preventDefault();
            // Используем новую архитектуру вместо старой
            applyParamsFromForm();
            return false;
        };
    }
});

// --- 3D визуализация Conv2d слоя с улучшенной логикой ---
function renderConvTensor3D(container) {
    container.innerHTML = '';
    const conv = window._state.conv;
    const data = conv.data;
    const channels = conv.channels;
    const height = data[0].length;
    const width = data[0][0].length;
    

    
    // Фиксированная область для информации о состоянии
    const statusArea = document.createElement('div');
    statusArea.style.height = '50px'; // Увеличенная фиксированная высота для предотвращения сдвигов
    statusArea.style.marginBottom = '10px';
    statusArea.style.display = 'flex';
    statusArea.style.justifyContent = 'center';
    statusArea.style.alignItems = 'center';
    statusArea.style.minHeight = '50px'; // Минимальная высота для стабильности
    
    if (window._highlight && window._highlight.fixed && window._highlight.layer === 1) {
        // Показываем компактную информацию о зафиксированном канале
        const fixedBadge = document.createElement('span');
        fixedBadge.style.background = '#ff5722';
        fixedBadge.style.color = '#fff';
        fixedBadge.style.padding = '6px 12px';
        fixedBadge.style.borderRadius = '16px';
        fixedBadge.style.fontWeight = 'bold';
        fixedBadge.style.fontSize = '13px';
        fixedBadge.textContent = `Канал ${window._highlight.channel + 1} • [${window._highlight.i + 1}, ${window._highlight.j + 1}]`;
        statusArea.appendChild(fixedBadge);
    } else {
        // Обычный селектор каналов
        renderChannelSelector(statusArea, channels, conv.selectedChannel, (idx) => {
            conv.selectedChannel = idx;
            window._needsFullRender = true; // Смена канала требует полного перерендера
            updateHighlightStyles();
        });
    }
    
    container.appendChild(statusArea);
    
    // Проверяем, есть ли активное выделение числа в более поздних Conv2d
    let highlightSet = new Set();
    let highlightMode = false;
    if (window._highlight && window._highlight.layer !== null && window._highlight.layer > 1) {
        // Получаем все влияющие ячейки через всю цепочку
        const allInfluencing = getAllInfluencingCells(
            window._highlight.layer, 
            window._highlight.channel, 
            window._highlight.i, 
            window._highlight.j
        );
        
        // Для единственного Conv2d слоя
        if (allInfluencing.has(1)) {
            highlightSet = allInfluencing.get(1);
        }
        highlightMode = true;
    }
    
    // Куб (все пластины) - фиксированные размеры для предотвращения дёрганья
    const cubeDiv = document.createElement('div');
    cubeDiv.style.position = 'relative';
    cubeDiv.style.width = `${width * 32}px`; // Убираем добавочную ширину от каналов для правильного выравнивания
    cubeDiv.style.height = `${height * 32}px`; // Убираем добавочную высоту от каналов
    cubeDiv.style.margin = '0 auto';
    cubeDiv.style.display = 'block';
    cubeDiv.style.minWidth = `${width * 32}px`;
    cubeDiv.style.minHeight = `${height * 32}px`;
    
    // Показываем только активный канал для предотвращения перекрытий
    for (let c = 0; c < channels; c++) {
        // Создаём слой только для активного канала или если есть выделение
        const shouldShowLayer = (c === conv.selectedChannel) || 
                               (highlightMode && highlightSet.size > 0) ||
                               (window._highlight && window._highlight.fixed && window._highlight.channel === c);
        
        if (!shouldShowLayer) continue;
        
        const layerDiv = document.createElement('div');
        layerDiv.style.position = 'absolute';
        // Убираем 3D смещение для активного канала, чтобы избежать перекрытий
        layerDiv.style.left = `0px`;
        layerDiv.style.top = `0px`;
        layerDiv.style.opacity = '0.95';
        layerDiv.style.background = `rgba(${100 + c*50},${200-c*60},255,0.18)`;
        layerDiv.style.border = '1.5px solid #bbb';
        layerDiv.style.borderRadius = '6px';
        layerDiv.style.width = `${width * 32}px`;
        layerDiv.style.height = `${height * 32}px`;
        layerDiv.style.transition = 'box-shadow 0.2s, background 0.2s';
        layerDiv.className = 'conv-cube-layer';
        layerDiv.dataset.channel = c;
        layerDiv.dataset.layer = 0; // Единственный слой имеет индекс 0
        layerDiv.style.pointerEvents = 'auto'; // ИСПРАВЛЕНО: разрешаем события мыши
        layerDiv.style.zIndex = 1000; // Высокий z-index для кликабельности
        
        if (highlightMode) {
            // Режим куба без чисел: показываем только влияющие числа
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const key = `${c}|${i}|${j}`;
                    if (highlightSet.has(key)) {
                        const cell = document.createElement('div');
                        cell.className = 'cell';
                        cell.textContent = data[c][i][j];
                        cell.style.position = 'absolute';
                        cell.style.left = `${j * 32}px`;
                        cell.style.top = `${i * 32}px`;
                        cell.style.width = '32px';
                        cell.style.height = '32px';
                        cell.style.display = 'flex';
                        cell.style.alignItems = 'center';
                        cell.style.justifyContent = 'center';
                        cell.style.fontSize = '15px';
                        cell.style.background = '#ffe066'; // жёлтый для влияющих
                        cell.style.border = '1px solid #ccc';
                        
                        // Добавляем data-атрибуты для кликабельности
                        cell.dataset.layer = 1;
                        cell.dataset.channel = c;
                        cell.dataset.i = i;
                        cell.dataset.j = j;
                        cell.style.cursor = 'pointer';
                        cell.style.pointerEvents = 'auto';
                        
                        layerDiv.appendChild(cell);
                    }
                }
            }
        } else {
            // Нормальный режим: создаём ячейки только для текущего канала
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.textContent = data[c][i][j];
                    cell.style.position = 'absolute';
                    cell.style.left = `${j * 32}px`;
                    cell.style.top = `${i * 32}px`;
                    cell.style.width = '32px';
                    cell.style.height = '32px';
                    cell.style.display = 'flex';
                    cell.style.alignItems = 'center';
                    cell.style.justifyContent = 'center';
                    cell.style.fontSize = '15px';
                    cell.style.opacity = '1';
                    
                    // Добавляем data-атрибуты для делегирования событий
                    cell.dataset.layer = 1;
                    cell.dataset.channel = c;
                    cell.dataset.i = i;
                    cell.dataset.j = j;
                        

                        


                        // Определяем цвет фона в зависимости от состояния
                        let backgroundColor = '#fff';
                        if (isHighlighted(1, c, i, j)) {
                            backgroundColor = '#ffe066'; // жёлтый для подсвеченных
                        }
                        if (window._highlight.fixed && 
                            window._highlight.layer === 1 && 
                            window._highlight.channel === c && 
                            window._highlight.i === i && 
                            window._highlight.j === j) {
                            backgroundColor = '#ff5722'; // красно-оранжевый для зафиксированных
                            cell.style.color = '#fff';
                            cell.style.fontWeight = 'bold';
                        }
                        cell.style.background = backgroundColor;
                        cell.style.border = '2px solid #666'; // Более заметная граница для кликабельных элементов
                        cell.style.cursor = 'pointer';
                        cell.style.pointerEvents = 'auto';
                        cell.style.zIndex = 1000; // Увеличиваем z-index для лучшей кликабельности
                        cell.style.transition = 'all 0.2s ease';
                        cell.style.position = 'absolute'; // Убеждаемся что position установлен
                        cell.style.userSelect = 'none'; // Предотвращаем выделение текста
                        
                        // Обработчики событий только для видимых ячеек
                        if (c === conv.selectedChannel) {
                            // Добавляем hover-эффект
                            cell.addEventListener('mouseenter', () => {
                                if (!window._highlight.fixed || 
                                    !(window._highlight.layer === 1 && 
                                      window._highlight.channel === c && 
                                      window._highlight.i === i && 
                                      window._highlight.j === j)) {
                                    cell.style.transform = 'scale(1.05)';
                                    cell.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
                                }
                            });
                            
                            cell.addEventListener('mouseleave', () => {
                                if (!window._highlight.fixed || 
                                    !(window._highlight.layer === 1 && 
                                      window._highlight.channel === c && 
                                      window._highlight.i === i && 
                                      window._highlight.j === j)) {
                                    cell.style.transform = 'scale(1)';
                                    cell.style.boxShadow = 'none';
                                }
                            });
                            
                            // Обработчик кликов
                            cell.addEventListener('click', (event) => {
                                event.stopPropagation();
                                window._clickInProgress = true;
                                
                                if (window._highlight.layer === 1 && 
                                    window._highlight.channel === c && 
                                    window._highlight.i === i && 
                                    window._highlight.j === j) {
                                    // Переключаем фиксацию
                                    window._highlight.fixed = !window._highlight.fixed;
                                    if (!window._highlight.fixed) {
                                        window._kernelShowChannel = {};
                                        window._state.conv.selectedKernelViewChannel = 0;
                                        // Полностью сбрасываем выделение при снятии фиксации
                                        window._highlight = {layer: null, channel: null, i: null, j: null, fixed: false};
                                    }
                                } else {
                                    // Устанавливаем новое выделение и фиксируем его
                                    window._highlight = {layer: 1, channel: c, i, j, fixed: true};
                                    window._kernelShowChannel = {};
                                    // Синхронизируем каналы при фиксации
                                    window._state.input.selectedChannel = 0;
                                    window._state.conv.selectedKernelViewChannel = 0;
                                }
                                
                                // Задержка перед перерендером
                                setTimeout(() => {
                                    renderAll();
                                    window._clickInProgress = false;
                                }, 10);
                            });
                        }
                    
                    layerDiv.appendChild(cell);
                }
            }
        }
        cubeDiv.appendChild(layerDiv);
    }
    container.appendChild(cubeDiv);
}

// --- Визуализация 4D ядра свёртки как 3D куба ---
function renderKernel4D(container) {
    container.innerHTML = '';
    const conv = window._state.conv;
    const kernel4D = conv.kernelWeights;
    
    if (!kernel4D) {
        container.innerHTML = '<div>Ядро не инициализировано</div>';
        return;
    }
    
    // Показываем ядро только при зафиксированном выделении
    if (!window._highlight || !window._highlight.fixed || window._highlight.layer !== 1) {
        const placeholder = document.createElement('div');
        placeholder.textContent = 'Кликните на выходную ячейку для просмотра ядра';
        placeholder.style.textAlign = 'center';
        placeholder.style.color = '#666';
        placeholder.style.fontStyle = 'italic';
        placeholder.style.padding = '20px';
        placeholder.style.height = '200px'; // Фиксированная высота для предотвращения сдвигов
        placeholder.style.minHeight = '200px';
        placeholder.style.display = 'flex';
        placeholder.style.alignItems = 'center';
        placeholder.style.justifyContent = 'center';
        container.appendChild(placeholder);
        return;
    }
    
    const outChannels = kernel4D.length;
    const inChannels = kernel4D[0].length;
    const kernelHeight = kernel4D[0][0].length;
    const kernelWidth = kernel4D[0][0][0].length;
    
    // Автоматическая синхронизация с выделенным выходным каналом
    let selectedOutChannel = conv.selectedKernelOutChannel;
    if (window._highlight && window._highlight.layer === 1) {
        selectedOutChannel = window._highlight.channel;
        conv.selectedKernelOutChannel = selectedOutChannel;
    }
    
    // Заголовок
    const title = document.createElement('div');
    title.textContent = `Ядро K[${selectedOutChannel + 1},:,:,:] (${inChannels}×${kernelHeight}×${kernelWidth})`;
    title.style.fontWeight = 'bold';
    title.style.marginBottom = '10px';
    title.style.textAlign = 'center';
    title.style.color = '#ff9800';
    title.style.width = '100%';
    title.style.height = '30px'; // Фиксированная высота
    title.style.minHeight = '30px';
    title.style.display = 'flex';
    title.style.alignItems = 'center';
    title.style.justifyContent = 'center';
    container.appendChild(title);
    
    // Определяем режим показа
    let isFixedMode = false;
    
    if (window._highlight && window._highlight.layer === 1 && window._highlight.fixed) {
        // Фиксированный режим - показываем всё ядро для конкретной ячейки
        isFixedMode = true;
        
        // Отображаем информацию о текущем канале (селектор перенесен в входной тензор)
        const channelInfo = document.createElement('div');
        channelInfo.style.textAlign = 'center';
        channelInfo.style.marginBottom = '15px';
        channelInfo.style.fontSize = '14px';
        channelInfo.style.color = '#ff5722';
        channelInfo.style.fontWeight = 'bold';
        channelInfo.style.padding = '8px';
        channelInfo.style.background = '#fff3e0';
        channelInfo.style.borderRadius = '4px';
        channelInfo.style.border = '1px solid #ff9800';
        channelInfo.style.height = '40px'; // Фиксированная высота
        channelInfo.style.minHeight = '40px';
        channelInfo.style.display = 'flex';
        channelInfo.style.alignItems = 'center';
        channelInfo.style.justifyContent = 'center';
        channelInfo.textContent = `Входной канал ядра: ${conv.selectedKernelViewChannel + 1} из ${inChannels}`;
        container.appendChild(channelInfo);
    }
    
    // 3D визуализация ядра
    const cubeDiv = document.createElement('div');
    cubeDiv.style.position = 'relative';
    cubeDiv.style.width = `${kernelWidth * 35 + inChannels * 15}px`;
    cubeDiv.style.height = `${kernelHeight * 35 + inChannels * 15}px`;
    cubeDiv.style.margin = '0 auto';
    cubeDiv.style.display = 'block';
    cubeDiv.style.cursor = 'pointer';
    
    for (let ic = 0; ic < inChannels; ic++) {
        const layerDiv = document.createElement('div');
        layerDiv.style.position = 'absolute';
        layerDiv.style.left = `${ic * 15}px`;
        layerDiv.style.top = `${ic * 15}px`;
        
        // Определяем активность слоя
        const isActiveLayer = isFixedMode ? (ic === conv.selectedKernelViewChannel) : true;
        
        // Устанавливаем прозрачность в зависимости от активности
        layerDiv.style.opacity = isActiveLayer ? '0.9' : '0.3';
        layerDiv.style.background = `rgba(255, 152, 0, ${isActiveLayer ? (0.1 + ic * 0.1) : 0.05})`;
        layerDiv.style.border = `2px solid ${isActiveLayer ? '#ff9800' : '#ffcc80'}`;
        layerDiv.style.borderRadius = '6px';
        layerDiv.style.width = `${kernelWidth * 35}px`;
        layerDiv.style.height = `${kernelHeight * 35}px`;
        layerDiv.style.zIndex = isActiveLayer ? ic + 100 : ic;
        layerDiv.className = 'kernel-cube-layer';
        layerDiv.dataset.channel = ic;
        layerDiv.dataset.layerIdx = 0;
        layerDiv.style.pointerEvents = 'none';
        layerDiv.style.transition = 'all 0.3s ease';
        
        // Определяем нужно ли показывать этот слой
        let shouldShowLayer = true; // Показываем все слои, но с разной прозрачностью
        
        if (shouldShowLayer) {
            const kernelMatrix = kernel4D[selectedOutChannel][ic];
            for (let h = 0; h < kernelHeight; h++) {
                for (let w = 0; w < kernelWidth; w++) {
                    const cell = document.createElement('div');
                    cell.textContent = kernelMatrix[h][w];
                    cell.style.position = 'absolute';
                    cell.style.left = `${w * 35}px`;
                    cell.style.top = `${h * 35}px`;
                    cell.style.width = '35px';
                    cell.style.height = '35px';
                    cell.style.display = 'flex';
                    cell.style.alignItems = 'center';
                    cell.style.justifyContent = 'center';
                    cell.style.fontSize = '13px';
                    cell.style.fontWeight = 'bold';
                    cell.style.border = '1px solid #ff9800';
                    cell.style.borderRadius = '3px';
                    cell.style.zIndex = 100;
                    
                    // Добавляем transition для плавных анимаций
                    cell.style.transition = 'all 0.2s ease';
                    cell.style.cursor = 'pointer';
                    
                    // В фиксированном режиме всё ядро участвует в свёртке
                    if (isFixedMode) {
                        cell.style.background = '#ff5722'; // красно-оранжевый для всего ядра при фиксации
                        cell.style.color = '#fff';
                        cell.style.boxShadow = '0 0 8px rgba(255, 87, 34, 0.8)';
                        cell.style.transform = 'scale(1.05)';
                        cell.style.fontWeight = 'bold';
                    } else {
                        // В обычном режиме стандартный вид
                        cell.style.background = '#fff3e0'; // светло-оранжевый фон
                        cell.style.color = '#e65100';
                    }
                    
                    // Добавляем hover-эффекты только для активных слоёв
                    if (isActiveLayer) {
                        cell.addEventListener('mouseenter', () => {
                            if (!isFixedMode) {
                                cell.style.background = '#ffcc02';
                                cell.style.color = '#bf360c';
                                cell.style.transform = 'scale(1.1)';
                                cell.style.boxShadow = '0 4px 12px rgba(255, 152, 0, 0.6)';
                                cell.style.zIndex = '200';
                                cell.style.fontWeight = 'bold';
                            }
                        });
                        
                        cell.addEventListener('mouseleave', () => {
                            if (!isFixedMode) {
                                cell.style.background = '#fff3e0';
                                cell.style.color = '#e65100';
                                cell.style.transform = 'scale(1)';
                                cell.style.boxShadow = 'none';
                                cell.style.zIndex = '100';
                                cell.style.fontWeight = 'normal';
                            }
                        });
                    }
                    
                    layerDiv.appendChild(cell);
                }
            }
        }
        
        cubeDiv.appendChild(layerDiv);
    }
    
    container.appendChild(cubeDiv);
    
    // Описание
    const description = document.createElement('div');
    description.style.textAlign = 'center';
    description.style.marginTop = '10px';
    description.style.fontSize = '12px';
    description.style.color = '#666';
    description.style.height = '40px'; // Фиксированная высота
    description.style.minHeight = '40px';
    description.style.display = 'flex';
    description.style.alignItems = 'center';
    description.style.justifyContent = 'center';
    
    if (isFixedMode) {
        description.innerHTML = `Ядро для вычисления ячейки <strong>Y[${selectedOutChannel + 1}][${window._highlight.i + 1}][${window._highlight.j + 1}]</strong>`;
        description.style.color = '#ff5722';
        description.style.fontWeight = 'normal';
        
        // УБРАНО: Блок отображения вычисления скалярного произведения
        // теперь отображается в отдельной области справа
    } else if (window._highlight && window._highlight.layer === 1) {
        description.textContent = `Все элементы ядра для канала ${selectedOutChannel + 1} (кликните для фиксации)`;
    } else {
        description.textContent = `3D ядро для выходного канала ${selectedOutChannel + 1}`;
    }
    container.appendChild(description);
}



// --- Проверка влияет ли конкретная ячейка входного тензора на зафиксированную выходную ячейку ---
function isInputCellInfluencing(inputChannel, paddedI, paddedJ) {
    if (!window._highlight || !window._highlight.fixed || window._highlight.layer !== 1) {
        return false;
    }
    
    const conv = window._state.conv;
    const stride = conv.stride;
    const kernelSize = conv.kernel;
    const paddingSize = conv.paddingSize || 0;
    
    const outputI = window._highlight.i;
    const outputJ = window._highlight.j;
    const outputChannel = window._highlight.channel;
    
    // Проверяем попадает ли ячейка входного тензора в рецептивное поле выходной ячейки
    for (let kh = 0; kh < kernelSize; kh++) {
        for (let kw = 0; kw < kernelSize; kw++) {
            const requiredPaddedI = outputI * stride + kh;
            const requiredPaddedJ = outputJ * stride + kw;
            
            if (requiredPaddedI === paddedI && requiredPaddedJ === paddedJ) {
                return true;
            }
        }
    }
    
    return false;
}

// --- Генерация математического представления вычисления свёртки ---
function generateConvolutionCalculation(outputChannel, outputI, outputJ) {
    const conv = window._state.conv;
    const input = window._state.input;
    const stride = conv.stride;
    const kernelSize = conv.kernel;
    const paddingType = conv.padding || 'valid';
    const paddingSize = conv.paddingSize || 0;
    const kernel4D = conv.kernelWeights;
    
    if (!kernel4D) return 'Ядро не инициализировано';
    
    // Применяем padding к входному тензору для получения значений
    const paddedInput = applyPadding(input.data, paddingType, paddingSize);
    const paddedHeight = paddedInput[0].length;
    const paddedWidth = paddedInput[0][0].length;
    
    const terms = [];
    let totalSum = 0;
    let totalParticipatingElements = 0; // Счётчик реально участвующих элементов
    
    // Для каждого входного канала
    for (let ic = 0; ic < input.channels; ic++) {
        let channelSum = 0;
        const channelTerms = [];
        
        // Для каждого элемента ядра - проверяем только те, которые реально участвуют
        for (let kh = 0; kh < kernelSize; kh++) {
            for (let kw = 0; kw < kernelSize; kw++) {
                // Вычисляем координаты в расширенном тензоре
                const paddedInputI = outputI * stride + kh;
                const paddedInputJ = outputJ * stride + kw;
                
                // ИСПРАВЛЕНИЕ: Проверяем границы расширенного тензора И что элемент реально участвует
                if (paddedInputI >= 0 && paddedInputI < paddedHeight && 
                    paddedInputJ >= 0 && paddedInputJ < paddedWidth) {
                    
                    const inputValue = paddedInput[ic][paddedInputI][paddedInputJ];
                    const kernelValue = kernel4D[outputChannel][ic][kh][kw];
                    const product = inputValue * kernelValue;
                    
                    // Добавляем только если элемент реально участвует в вычислении
                    channelSum += product;
                    totalParticipatingElements++;
                    
                    // Определяем координаты в исходном тензоре для отображения
                    const origI = paddedInputI - paddingSize;
                    const origJ = paddedInputJ - paddingSize;
                    const isPadding = origI < 0 || origI >= input.height || origJ < 0 || origJ >= input.width;
                    
                    let inputCoord;
                    if (isPadding) {
                        inputCoord = `<span style="color: #ff9800; font-style: italic;">pad[${paddedInputI + 1},${paddedInputJ + 1}]</span>`;
                    } else {
                        inputCoord = `X[${ic + 1}][${origI + 1}][${origJ + 1}]`;
                    }
                    
                    const color = product >= 0 ? '#2e7d32' : '#c62828';
                    channelTerms.push(
                        `<span style="color: ${color};">${inputCoord}×K[${outputChannel + 1}][${ic + 1}][${kh + 1}][${kw + 1}] = ${inputValue}×${kernelValue} = ${product}</span>`
                    );
                }
                // Если элемент выходит за границы - он НЕ участвует в вычислении, не добавляем его
            }
        }
        
        // Добавляем канал только если есть реально участвующие элементы
        if (channelTerms.length > 0) {
            totalSum += channelSum;
            terms.push({
                channel: ic,
                sum: channelSum,
                terms: channelTerms
            });
        }
    }
    
    // Формируем заголовок с информацией о количестве участвующих элементов
    const totalPossibleElements = input.channels * kernelSize * kernelSize;
    let html = `<div style="color: #d84315; font-weight: bold; margin-bottom: 8px;">
        Вычисление Y[${outputChannel + 1}][${outputI + 1}][${outputJ + 1}]:
    </div>`;
    
    html += `<div style="margin-bottom: 8px; color: #5d4037;">
        Y[${outputChannel + 1}][${outputI + 1}][${outputJ + 1}] = ∑<sub>c</sub> ∑<sub>h</sub> ∑<sub>w</sub> 
        X[c][i*stride+h][j*stride+w] × K[${outputChannel + 1}][c][h][w]
    </div>`;
    

    
    // Показываем выбранный канал или все каналы
    const selectedKernelChannel = conv.selectedKernelViewChannel;
    const showAllChannels = !window._highlight.fixed;
    
    if (showAllChannels || terms.length === 1) {
        // Показываем все каналы
        for (const term of terms) {
            const channelColor = term.channel === selectedKernelChannel ? '#ff5722' : '#666';
            html += `<div style="margin: 8px 0; padding: 6px; background: #f5f5f5; border-left: 3px solid ${channelColor};">
                <div style="color: ${channelColor}; font-weight: bold;">Канал ${term.channel + 1}:</div>
                <div style="margin: 4px 0; font-size: 13px; line-height: 1.4;">
                    ${term.terms.join('<br>')}
                </div>
                <div style="color: #d84315; font-weight: bold; margin-top: 4px;">
                    Сумма канала ${term.channel + 1}: ${term.sum}
                </div>
            </div>`;
        }
        
        html += `<div style="margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #ff5722, #d84315); color: white; border-radius: 8px; font-weight: bold; text-align: center; font-size: 14px; box-shadow: 0 4px 8px rgba(255, 87, 34, 0.3); border: 2px solid #bf360c;">
            ОБЩИЙ РЕЗУЛЬТАТ: Y[${outputChannel + 1}][${outputI + 1}][${outputJ + 1}] = ${totalSum}
        </div>`;
    } else {
        // Показываем только выбранный канал
        const selectedTerm = terms.find(t => t.channel === selectedKernelChannel);
        if (selectedTerm) {
            html += `<div style="margin: 8px 0; padding: 6px; background: #fff3e0; border: 2px solid #ff5722;">
                <div style="color: #ff5722; font-weight: bold;">Канал ${selectedTerm.channel + 1} (выбранный):</div>
                <div style="margin: 4px 0; font-size: 13px; line-height: 1.4;">
                    ${selectedTerm.terms.join('<br>')}
                </div>
                <div style="color: #d84315; font-weight: bold; margin-top: 4px;">
                    Вклад канала ${selectedTerm.channel + 1}: ${selectedTerm.sum}
                </div>
            </div>`;
            
            html += `<div style="margin-top: 12px; padding: 10px; background: linear-gradient(135deg, #ff5722, #d84315); color: white; border-radius: 6px; font-weight: bold; text-align: center; font-size: 13px; box-shadow: 0 3px 6px rgba(255, 87, 34, 0.3);">
                ОБЩИЙ РЕЗУЛЬТАТ: ${totalSum}
            </div>`;
            
            html += `<div style="margin-top: 8px; color: #666; font-size: 12px; font-style: italic; text-align: center;">
                (сумма всех ${input.channels} каналов)
            </div>`;
        }
    }
    
    return html;
}

// --- Рекурсивный поиск всех влияющих ячеек через всю цепочку слоёв ---
function getAllInfluencingCells(layerIdx, channelIdx, i, j) {
    const result = new Map(); // layer -> Set of "channel|i|j"
    
    function collectInfluencing(layer, channel, row, col) {
        const key = `${layer}|${channel}|${row}|${col}`;
        if (!result.has(layer)) {
            result.set(layer, new Set());
        }
        result.get(layer).add(`${channel}|${row}|${col}`);
        
        // Если это не входной слой, ищем влияющие ячейки в предыдущем слое
        if (layer > 0) {
            const rf = getReceptiveFieldCoords(layer, channel, row, col);
            for (const pos of rf) {
                collectInfluencing(pos.layer, pos.channel, pos.i, pos.j);
            }
        }
    }
    
    collectInfluencing(layerIdx, channelIdx, i, j);
    return result;
}

// --- Функция для переключения состояния поля размера padding ---
function togglePaddingSize() {
    const paddingType = document.getElementById('paddingType').value;
    const paddingSizeInput = document.getElementById('paddingSize');
    const paddingSizeHint = document.getElementById('paddingSizeHint');
    
    if (paddingType === 'valid') {
        // Делаем поле неактивным и устанавливаем значение 0
        paddingSizeInput.disabled = true;
        paddingSizeInput.value = 0;
        paddingSizeInput.style.opacity = '0.5';
        paddingSizeInput.style.cursor = 'not-allowed';
        paddingSizeHint.textContent = 'Неактивно в режиме Valid (без рамки)';
        paddingSizeHint.style.color = '#999';
        paddingSizeHint.style.fontStyle = 'italic';
    } else {
        // Делаем поле активным
        paddingSizeInput.disabled = false;
        paddingSizeInput.style.opacity = '1';
        paddingSizeInput.style.cursor = 'text';
        paddingSizeHint.textContent = 'Должен быть меньше размера ядра';
        paddingSizeHint.style.color = '#666';
        paddingSizeHint.style.fontStyle = 'normal';
        // Если значение было 0, устанавливаем 1 по умолчанию
        if (paddingSizeInput.value === '0') {
            paddingSizeInput.value = '1';
        }
    }
}

// --- Обработка параметров из формы ---