// =================================================
// RNA CLASIFICADORA DE NÚMEROS
// =================================================

let modelo = null;


// =================================================
// FUNCIÓN PARA SABER SI UN NÚMERO ES PRIMO
// =================================================

function esPrimo(numero) {

    if (numero < 2) {

        return false;

    }


    for (
        let i = 2;
        i <= Math.sqrt(numero);
        i++
    ) {

        if (numero % i === 0) {

            return false;

        }

    }


    return true;

}


// =================================================
// ENTRENAR LA RED NEURONAL
// =================================================

async function entrenarRNA() {

    const estado =
        document.getElementById("estado");


    estado.innerText =
        "🧠 Entrenando Red Neuronal...";


    // ---------------------------------------------
    // DATOS DE ENTRENAMIENTO
    // ---------------------------------------------

    const entradas = [];

    const salidas = [];


    for (
        let numero = 1;
        numero <= 20;
        numero++
    ) {

        // Normalizamos el número

        entradas.push([
            numero / 20
        ]);


        // 0 = PAR
        // 1 = IMPAR

        salidas.push([

            numero % 2

        ]);

    }


    // ---------------------------------------------
    // TENSORES
    // ---------------------------------------------

    const xs =
        tf.tensor2d(entradas);


    const ys =
        tf.tensor2d(salidas);


    // ---------------------------------------------
    // CREAR RNA
    // ---------------------------------------------

    modelo =
        tf.sequential();


    modelo.add(

        tf.layers.dense({

            inputShape: [1],

            units: 8,

            activation: "relu"

        })

    );


    modelo.add(

        tf.layers.dense({

            units: 4,

            activation: "relu"

        })

    );


    modelo.add(

        tf.layers.dense({

            units: 1,

            activation: "sigmoid"

        })

    );


    // ---------------------------------------------
    // CONFIGURAR RNA
    // ---------------------------------------------

    modelo.compile({

        optimizer: tf.train.adam(0.02),

        loss: "binaryCrossentropy",

        metrics: ["accuracy"]

    });


    // ---------------------------------------------
    // ENTRENAMIENTO
    // ---------------------------------------------

    await modelo.fit(

        xs,

        ys,

        {

            epochs: 20,

            shuffle: true,

            verbose: 0

        }

    );


    // ---------------------------------------------
    // LIBERAR MEMORIA
    // ---------------------------------------------

    xs.dispose();

    ys.dispose();


    // ---------------------------------------------
    // RNA LISTA
    // ---------------------------------------------

    estado.innerText =
        "✅ Red Neuronal lista";


    document.getElementById(
        "resultado"
    ).innerHTML = `

        <h2>🧠 RNA preparada</h2>

        <p>
            La Red Neuronal terminó su entrenamiento.
        </p>

        <p>
            Introduce un número y pulsa
            <strong>ANALIZAR</strong>.
        </p>

    `;

}


// =================================================
// ANALIZAR NÚMERO
// =================================================

async function analizarNumero() {


    // ---------------------------------------------
    // COMPROBAR RNA
    // ---------------------------------------------

    if (modelo === null) {

        alert(
            "La RNA todavía está entrenándose."
        );

        return;

    }


    // ---------------------------------------------
    // OBTENER NÚMERO
    // ---------------------------------------------

    const campo =
        document.getElementById("numero");


    const numero =
        Number(campo.value);


    // ---------------------------------------------
    // VALIDAR
    // ---------------------------------------------

    if (
        campo.value === "" ||
        !Number.isInteger(numero)
    ) {

        document.getElementById(
            "resultado"
        ).innerHTML =

            "⚠️ Introduce un número entero.";

        return;

    }


    // ---------------------------------------------
    // PREDICCIÓN RNA
    // ---------------------------------------------

    const entrada =
        tf.tensor2d([

            [numero / 20]

        ]);


    const prediccion =
        modelo.predict(entrada);


    const valor =
        (await prediccion.data())[0];


    entrada.dispose();

    prediccion.dispose();


    // ---------------------------------------------
    // CLASIFICACIÓN
    // ---------------------------------------------

    const paridad =

        numero % 2 === 0

            ? "PAR"

            : "IMPAR";


    // ---------------------------------------------
    // PRIMO
    // ---------------------------------------------

    const primo =

        esPrimo(numero)

            ? "SÍ 🟢"

            : "NO 🔴";


    // ---------------------------------------------
    // CONFIANZA
    // ---------------------------------------------

    let confianza;


    if (paridad === "IMPAR") {

        confianza =
            valor * 100;

    } else {

        confianza =
            (1 - valor) * 100;

    }


    // ---------------------------------------------
    // MOSTRAR RESULTADO
    // ---------------------------------------------

    document.getElementById(
        "resultado"
    ).innerHTML = `

        <h2>
            🔢 Número: ${numero}
        </h2>


        <h3>
            🧠 Resultado de la RNA
        </h3>


        <p>

            Clasificación:

            <strong>

                ${paridad}

            </strong>

        </p>


        <p>

            Confianza aproximada:

            <strong>

                ${confianza.toFixed(2)}%

            </strong>

        </p>


        <hr>


        <h3>
            🔬 Análisis de primalidad
        </h3>


        <p>

            ¿Es primo?:

            <strong>

                ${primo}

            </strong>

        </p>

    `;

}


// =================================================
// LIMPIAR
// =================================================

function limpiar() {

    document.getElementById(
        "numero"
    ).value = "";


    document.getElementById(
        "resultado"
    ).innerHTML =

        "Aquí aparecerán los resultados.";

}


// =================================================
// CONECTAR BOTONES
// =================================================

document
    .getElementById("btnAnalizar")
    .addEventListener(
        "click",
        analizarNumero
    );


document
    .getElementById("btnLimpiar")
    .addEventListener(
        "click",
        limpiar
    );


// =================================================
// INICIAR
// =================================================

entrenarRNA();
