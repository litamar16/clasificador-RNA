let modeloParidad;
let modeloPrimo;


// ==========================================
// COMPROBACIÓN MATEMÁTICA DE PRIMO
// ==========================================

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


// ==========================================
// RNA 1
// PAR / IMPAR
// ==========================================

async function entrenarRNAParidad() {

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

  const xs = tf.tensor2d(entradas);
  const ys = tf.tensor2d(salidas);


  modeloParidad = tf.sequential();

  modeloParidad.add(
    tf.layers.dense({
      inputShape: [1],
      units: 16,
      activation: "relu"
    })
  );

  modeloParidad.add(
    tf.layers.dense({
      units: 8,
      activation: "relu"
    })
  );

  modeloParidad.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    })
  );


  modeloParidad.compile({

    optimizer: tf.train.adam(0.01),

    loss: "binaryCrossentropy",

    metrics: ["accuracy"]

  });


  await modeloParidad.fit(xs, ys, {

    epochs: 100,

    shuffle: true,

    verbose: 0

  });


  xs.dispose();
  ys.dispose();
}


// ==========================================
// RNA 2
// PRIMO / NO PRIMO
// ==========================================

async function entrenarRNAPrimos() {

  const entradas = [];
  const salidas = [];


  // Entrenamiento del 1 al 100

  for (let numero = 1; numero <= 100; numero++) {

    entradas.push([numero / 100]);


    if (esPrimo(numero)) {

      salidas.push([1]);

    } else {

      salidas.push([0]);

    }

  }


  const xs = tf.tensor2d(entradas);
  const ys = tf.tensor2d(salidas);


  modeloPrimo = tf.sequential();


  modeloPrimo.add(
    tf.layers.dense({
      inputShape: [1],
      units: 32,
      activation: "relu"
    })
  );


  modeloPrimo.add(
    tf.layers.dense({
      units: 16,
      activation: "relu"
    })
  );


  modeloPrimo.add(
    tf.layers.dense({
      units: 8,
      activation: "relu"
    })
  );


  modeloPrimo.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    })
  );


  modeloPrimo.compile({

    optimizer: tf.train.adam(0.01),

    loss: "binaryCrossentropy",

    metrics: ["accuracy"]

  });


  await modeloPrimo.fit(xs, ys, {

    epochs: 300,

    shuffle: true,

    verbose: 0

  });


  xs.dispose();
  ys.dispose();
}


// ==========================================
// ENTRENAR LAS DOS RNA
// ==========================================

async function entrenarTodo() {

  document.getElementById("estado").innerText =
    "Entrenando RNA 1...";


  await entrenarRNAParidad();


  document.getElementById("estado").innerText =
    "Entrenando RNA 2...";


  await entrenarRNAPrimos();


  document.getElementById("estado").innerText =
    "¡RNA 1 y RNA 2 listas! ✅";


  document.getElementById("resultado").innerHTML = `

    <h2>🧠 IA preparada</h2>

    <p>
      RNA 1: Par / Impar ✅
    </p>

    <p>
      RNA 2: Primo / No primo ✅
    </p>

    <p>
      Entrenamiento: números del 1 al 100
    </p>

  `;
}


// ==========================================
// ANALIZAR NÚMERO
// ==========================================

async function analizarNumero() {


  if (!modeloParidad || !modeloPrimo) {

    alert(
      "Las redes neuronales todavía se están entrenando."
    );

    return;

  }


  const numero = Number(
    document.getElementById("numero").value
  );


  if (
    document.getElementById("numero").value === ""
    || !Number.isInteger(numero)
  ) {

    document.getElementById("resultado").innerHTML =
      "⚠️ Introduce un número entero.";

    return;

  }


  if (numero < 1 || numero > 100) {

    document.getElementById("resultado").innerHTML = `

      <h3>⚠️ Número fuera del rango</h3>

      <p>
        Las RNA fueron entrenadas con números
        del <strong>1 al 100</strong>.
      </p>

    `;

    return;

  }


  // ==========================================
  // RNA 1: PAR / IMPAR
  // ==========================================

  const entradaParidad = tf.tensor2d([
    [numero / 100]
  ]);


  const prediccionParidad =
    modeloParidad.predict(entradaParidad);


  const valorParidad =
    (await prediccionParidad.data())[0];


  entradaParidad.dispose();
  prediccionParidad.dispose();


  let resultadoParidad;
  let confianzaParidad;


  if (valorParidad >= 0.5) {

    resultadoParidad = "IMPAR";

    confianzaParidad =
      valorParidad * 100;

  } else {

    resultadoParidad = "PAR";

    confianzaParidad =
      (1 - valorParidad) * 100;

  }


  // ==========================================
  // RNA 2: PRIMO / NO PRIMO
  // ==========================================

  const entradaPrimo = tf.tensor2d([
    [numero / 100]
  ]);


  const prediccionPrimo =
    modeloPrimo.predict(entradaPrimo);


  const valorPrimo =
    (await prediccionPrimo.data())[0];


  entradaPrimo.dispose();
  prediccionPrimo.dispose();


  let resultadoPrimo;
  let confianzaPrimo;


  if (valorPrimo >= 0.5) {

    resultadoPrimo = "PRIMO";

    confianzaPrimo =
      valorPrimo * 100;

  } else {

    resultadoPrimo = "NO PRIMO";

    confianzaPrimo =
      (1 - valorPrimo) * 100;

  }


  // ==========================================
  // COMPROBACIÓN MATEMÁTICA
  // ==========================================

  const comprobacionParidad =
    numero % 2 === 0
      ? "PAR"
      : "IMPAR";


  const comprobacionPrimo =
    esPrimo(numero)
      ? "PRIMO"
      : "NO PRIMO";


  // ==========================================
  // MOSTRAR RESULTADO
  // ==========================================

  document.getElementById("resultado").innerHTML = `

    <h2>🔢 Número: ${numero}</h2>

    <hr>

    <h3>🧠 RNA 1 — Paridad</h3>

    <p>
      Predicción:
      <strong>${resultadoParidad}</strong>
    </p>

    <p>
      Confianza:
      <strong>
        ${confianzaParidad.toFixed(2)}%
      </strong>
    </p>


    <hr>


    <h3>🧠 RNA 2 — Primalidad</h3>

    <p>
      Predicción:
      <strong>${resultadoPrimo}</strong>
    </p>

    <p>
      Confianza:
      <strong>
        ${confianzaPrimo.toFixed(2)}%
      </strong>
    </p>


    <hr>


    <h3>🔬 Comprobación matemática</h3>

    <p>
      Paridad:
      <strong>${comprobacionParidad}</strong>
    </p>

    <p>
      Número primo:
      <strong>${comprobacionPrimo}</strong>
    </p>

  `;
}


// ==========================================
// INICIAR ENTRENAMIENTO
// ==========================================

entrenarTodo();
