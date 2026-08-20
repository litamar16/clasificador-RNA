
function esPrimo(numero) {

  if (numero < 2) {
    return false;
  }

  for (let i = 2; i <= Math.sqrt(numero); i++) {
    if (numero % i === 0) {
      return false;
    }
  }

  return true;
}


function analizarNumero() {

  let numero = Number(
    document.getElementById("numero").value
  );

  if (isNaN(numero)) {
    document.getElementById("resultado").innerHTML =
      "⚠️ Introduce un número.";
    return;
  }

  let paridad;

  if (numero % 2 === 0) {
    paridad = "PAR";
  } else {
    paridad = "IMPAR";
  }

  let primo = esPrimo(numero);

  document.getElementById("resultado").innerHTML =
    `
    <strong>🔢 Número:</strong> ${numero}<br><br>

    <strong>📊 Clasificación:</strong> ${paridad}<br>

    <strong>🔬 Primo:</strong>
    ${primo ? "SÍ 🟢" : "NO 🔴"}
    `;
}



let modelo;

// Crear y entrenar la RNA
async function entrenarRNA() {

  document.getElementById("estado").innerText =
    "Entrenando...";

  /*
    Datos de entrenamiento.

    0 = PAR
    1 = IMPAR
  */

  const entradas = [];
  const salidas = [];

  for (let numero = 0; numero <= 100; numero++) {

    entradas.push([numero / 100]);

    if (numero % 2 === 0) {
      salidas.push([0]);
    } else {
      salidas.push([1]);
    }
  }

  // Convertir datos a tensores
  const xs = tf.tensor2d(entradas);
  const ys = tf.tensor2d(salidas);

  // Crear la Red Neuronal
  modelo = tf.sequential();

  // Primera capa
  modelo.add(
    tf.layers.dense({
      inputShape: [1],
      units: 16,
      activation: "relu"
    })
  );

  // Segunda capa
  modelo.add(
    tf.layers.dense({
      units: 8,
      activation: "relu"
    })
  );

  // Capa de salida
  modelo.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    })
  );

  // Configurar entrenamiento
  modelo.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  // Entrenar
  await modelo.fit(xs, ys, {
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoca, logs) => {

        document.getElementById("estado").innerText =
          "Entrenando: " + (epoca + 1) + "%";
      }
    }
  });

  // Liberar memoria
  xs.dispose();
  ys.dispose();

  document.getElementById("estado").innerText =
    "RNA entrenada ✅";

  document.getElementById("resultado").innerHTML =
    "🧠 La Red Neuronal está lista.";
}


// Analizar número
async function analizarNumero() {

  if (!modelo) {

    alert("La RNA todavía está entrenándose.");

    return;
  }

  const numero =
    Number(document.getElementById("numero").value);

  if (isNaN(numero)) {

    alert("Introduce un número.");

    return;
  }

  // Preparar número
  const entrada = tf.tensor2d([
    [numero / 100]
  ]);

  // Realizar predicción
  const prediccion =
    modelo.predict(entrada);

  const valor =
    (await prediccion.data())[0];

  entrada.dispose();
  prediccion.dispose();

  let resultado;
  let confianza;

  if (valor >= 0.5) {

    resultado = "IMPAR";
    confianza = valor * 100;

  } else {

    resultado = "PAR";
    confianza = (1 - valor) * 100;

  }

  // Mostrar resultado
  document.getElementById("resultado").innerHTML = `
    
    <h2>🔢 ${numero}</h2>

    <p>
      🧠 <strong>Predicción de la RNA:</strong>
    </p>

    <h2>${resultado}</h2>

    <p>
      📊 Confianza:
      <strong>${confianza.toFixed(2)}%</strong>
    </p>

  `;
}


// Entrenar automáticamente
entrenarRNA();
