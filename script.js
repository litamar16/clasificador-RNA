let modelo = null;


// ==========================================
// DATOS PARA LA RED NEURONAL
// ==========================================

// La RNA recibe características del número:
//
// 1. Número normalizado
// 2. Divisible entre 2
// 3. Divisible entre 3
// 4. Divisible entre 5
// 5. Divisible entre 7


function caracteristicas(n) {

  return [
    n / 100,
    n % 2 === 0 ? 1 : 0,
    n % 3 === 0 ? 1 : 0,
    n % 5 === 0 ? 1 : 0,
    n % 7 === 0 ? 1 : 0
  ];

}


// ==========================================
// SABER SI ES PRIMO
// ==========================================

function esPrimo(n) {

  if (n < 2) {
    return false;
  }

  for (let i = 2; i <= Math.sqrt(n); i++) {

    if (n % i === 0) {
      return false;
    }

  }

  return true;

}


// ==========================================
// ENTRENAR RNA
// ==========================================

async function entrenarRNA() {

  const estado =
    document.getElementById("estado");

  estado.innerText =
    "🧠 Entrenando RNA...";


  const entradas = [];
  const salidas = [];


  // Crear ejemplos del 1 al 100

  for (let n = 1; n <= 100; n++) {

    entradas.push(
      caracteristicas(n)
    );


    // Primera salida:
    // 0 = PAR
    // 1 = IMPAR
    //
    // Segunda salida:
    // 0 = NO PRIMO
    // 1 = PRIMO

    salidas.push([

      n % 2 === 0 ? 0 : 1,

      esPrimo(n) ? 1 : 0

    ]);

  }


  const xs =
    tf.tensor2d(entradas);


  const ys =
    tf.tensor2d(salidas);


  // ========================================
  // CREAR RED
  // ========================================

  modelo =
    tf.sequential();


  // Capa de entrada + neuronas ocultas

  modelo.add(
    tf.layers.dense({

      inputShape: [5],

      units: 12,

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


  // Dos salidas:
  //
  // salida 1 = par/impar
  // salida 2 = primo/no primo

  modelo.add(
    tf.layers.dense({

      units: 2,

      activation: "sigmoid"

    })
  );


  modelo.compile({

    optimizer: tf.train.adam(0.03),

    loss: "binaryCrossentropy",

    metrics: ["accuracy"]

  });


  // ========================================
  // ENTRENAMIENTO
  // ========================================

  await modelo.fit(

    xs,

    ys,

    {

      epochs: 25,

      shuffle: true,

      verbose: 0

    }

  );


  // Liberar memoria

  xs.dispose();

  ys.dispose();


  estado.innerText =
    "🧠 RNA lista ✅";


  document.getElementById(
    "resultado"
  ).innerHTML = `

    <h2>🧠 Red Neuronal lista</h2>

    <p>
      La RNA ha terminado su entrenamiento.
    </p>

    <p>
      Puede clasificar:
    </p>

    <p>
      🔵 PAR / IMPAR
    </p>

    <p>
      🟢 PRIMO / NO PRIMO
    </p>

  `;

}


// ==========================================
// ANALIZAR NÚMERO
// ==========================================

async function analizarNumero() {

  if (!modelo) {

    alert(
      "Espera a que la RNA termine de entrenarse."
    );

    return;

  }


  const numero =
    Number(
      document.getElementById(
        "numero"
      ).value
    );


  if (
    !Number.isInteger(numero)
  ) {

    document.getElementById(
      "resultado"
    ).innerHTML =

      "⚠️ Introduce un número entero.";

    return;

  }


  if (
    numero < 1 ||
    numero > 100
  ) {

    document.getElementById(
      "resultado"
    ).innerHTML =

      "⚠️ Introduce un número entre 1 y 100.";

    return;

  }


  // ========================================
  // PREPARAR NÚMERO
  // ========================================

  const entrada =
    tf.tensor2d([

      caracteristicas(numero)

    ]);


  // ========================================
  // PREDICCIÓN DE LA RNA
  // ========================================

  const prediccion =
    modelo.predict(entrada);


  const valores =
    await prediccion.array();


  entrada.dispose();

  prediccion.dispose();


  // Obtener las dos predicciones

  const valorParidad =
    valores[0][0];


  const valorPrimo =
    valores[0][1];


  // ========================================
  // RESULTADO RNA
  // ========================================

  const resultadoParidad =
    valorParidad >= 0.5
      ? "IMPAR"
      : "PAR";


  const confianzaParidad =
    valorParidad >= 0.5
      ? valorParidad * 100
      : (1 - valorParidad) * 100;


  const resultadoPrimo =
    valorPrimo >= 0.5
      ? "PRIMO"
      : "NO PRIMO";


  const confianzaPrimo =
    valorPrimo >= 0.5
      ? valorPrimo * 100
      : (1 - valorPrimo) * 100;


  // ========================================
  // COMPROBACIÓN MATEMÁTICA
  // ========================================

  const resultadoRealParidad =
    numero % 2 === 0
      ? "PAR"
      : "IMPAR";


  const resultadoRealPrimo =
    esPrimo(numero)
      ? "PRIMO"
      : "NO PRIMO";


  // ========================================
  // MOSTRAR RESULTADO
  // ========================================

  document.getElementById(
    "resultado"
  ).innerHTML = `

    <h2>
      🔢 Número: ${numero}
    </h2>

    <hr>

    <h3>
      🧠 RNA 1 — PAR / IMPAR
    </h3>

    <p>
      Predicción:
      <strong>
        ${resultadoParidad}
      </strong>
    </p>

    <p>
      Confianza:
      <strong>
        ${confianzaParidad.toFixed(2)}%
      </strong>
    </p>

    <hr>

    <h3>
      🧠 RNA 2 — PRIMO / NO PRIMO
    </h3>

    <p>
      Predicción:
      <strong>
        ${resultadoPrimo}
      </strong>
    </p>

    <p>
      Confianza:
      <strong>
        ${confianzaPrimo.toFixed(2)}%
      </strong>
    </p>

    <hr>

    <h3>
      🔬 Comprobación matemática
    </h3>

    <p>
      Paridad real:
      <strong>
        ${resultadoRealParidad}
      </strong>
    </p>

    <p>
      Resultado real:
      <strong>
        ${resultadoRealPrimo}
      </strong>
    </p>

  `;

}


// ==========================================
// LIMPIAR
// ==========================================

function limpiar() {

  document.getElementById(
    "numero"
  ).value = "";


  document.getElementById(
    "resultado"
  ).innerHTML = "";

}


// ==========================================
// INICIAR RNA
// ==========================================

entrenarRNA();
